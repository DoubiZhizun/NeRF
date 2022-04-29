package nerf

import ai.djl.engine._
import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.ndarray.types._
import ai.djl.training.loss._

class nerf(config: nerfConfig, manager: NDManager) {

  val loss = Loss.l2Loss("L2Loss", 1)
  val coarseBlock = if (config.NImportance == 0) null else new coreBlock(config).initialize(manager)
  val fineBlock = new coreBlock(config).initialize(manager)

  def predict(raysO: NDArray, raysD: NDArray, time: NDArray, near: NDArray, far: NDArray, viewdir: NDArray): NDArray = {
    val (_, fineRgbOut) = forward(raysO, raysD, time, near, far, viewdir, false)
    fineRgbOut
  }

  def train(raysO: NDArray, raysD: NDArray, time: NDArray, near: NDArray, far: NDArray, viewdir: NDArray, label: NDArray): Float = {
    //label：尺寸(batchNum, 3)
    val collector = Engine.getInstance().newGradientCollector()
    val (coarseRgbOut, fineRgbOut) = forward(raysO, raysD, time, near, far, viewdir, true)
    val lossValue = lossCalculate(coarseRgbOut, fineRgbOut, label)
    collector.backward(lossValue)
    collector.close()
    config.ps.updateAllParameters()
    lossValue.getFloat()
  }

  val forward = if (config.NImportance == 0) forwardNoCoarse _ else forwardWithCoarse _
  val lossCalculate = if (config.NImportance == 0) (coarse: NDArray, fine: NDArray, label: NDArray) => loss.evaluate(new NDList(label), new NDList(fine)) else (coarse: NDArray, fine: NDArray, label: NDArray) => loss.evaluate(new NDList(label), new NDList(coarse)).add(loss.evaluate(new NDList(label), new NDList(fine)))

  def forwardWithCoarse(raysO: NDArray, raysD: NDArray, time: NDArray, near: NDArray, far: NDArray, viewdir: NDArray, training: Boolean): (NDArray, NDArray) = {
    //输入在getInput中有详细介绍
    //viewdir：尺寸(batchNum, 1, 1, 3)
    val (coarsePos, coarseZVals) = getInput(raysO, raysD, near, far)
    val (coarseD, coarseRgb) = coarseBlock.forward(coarsePos, viewdir, time, training)
    val coarseWeight = getWeight(coarseD, coarseZVals)
    val (finePos, fineZVals) = samplePdf(coarseWeight, coarseZVals, raysO, raysD)
    val (findD, fineRgb) = fineBlock.forward(finePos, viewdir, time, training)
    val fineWeight = getWeight(findD, fineZVals)
    val fineRgbOut = addBkgd(fineWeight.mul(fineRgb.getNDArrayInternal.sigmoid()).sum(Array(1)), fineWeight)
    val coarseRgbOut = if (training) addBkgd(coarseWeight.mul(coarseRgb.getNDArrayInternal.sigmoid()).sum(Array(1)), coarseWeight) else null
    (coarseRgbOut, fineRgbOut)
    //输出：
    //coarseRgbOut：粗糙网络渲染结果，尺寸(batchNum, 3)
    //fineRgbOut：细腻网络渲染结果，尺寸(batchNum, 3)
  }

  def forwardNoCoarse(raysO: NDArray, raysD: NDArray, time: NDArray, near: NDArray, far: NDArray, viewdir: NDArray, training: Boolean): (NDArray, NDArray) = {
    val (finePos, fineZVals) = getInput(raysO, raysD, near, far)
    val (findD, fineRgb) = fineBlock.forward(finePos, viewdir, time, training)
    val fineWeight = getWeight(findD, fineZVals)
    val fineRgbOut = addBkgd(fineWeight.mul(fineRgb.getNDArrayInternal.sigmoid()).sum(Array(1)), fineWeight)
    (null, fineRgbOut)
    //no coarse
  }

  var addNoise = if (config.rawNoiseStd > 0) (input: NDArray) => input.add(input.getManager.randomNormal(input.getShape).mul(config.rawNoiseStd))
  else (input: NDArray) => input

  def noise(input: Boolean): Unit = {
    if (input && config.rawNoiseStd > 0) (input: NDArray) => input.add(input.getManager.randomNormal(input.getShape).mul(config.rawNoiseStd))
    else (input: NDArray) => input
  }

  val addBkgd = if (config.whiteBkgd) (rgb: NDArray, weight: NDArray) => {
    rgb.add(weight.sum(Array(1)).sub(1).neg())
    //对于这个函数的理解：
    //假设渲染出的所有点的颜色都是白色，计算最终渲染出的颜色，
    //然后用纯白色去减这个颜色，就能得到背景所需的颜色
  } else (rgb: NDArray, weight: NDArray) => rgb

  def getWeight(d: NDArray, zVals: NDArray): NDArray = {
    //将网络输出的密度计算成权重
    //d：尺寸(batchNum, NSamples ( + NImportance), 1)
    //zVals：尺寸(batchNum ,NSamples ( + NImportance), 1)
    val dists = zVals.get(":,1:,:").sub(zVals.get(":,:-1,:"))
    //raysD归一化过，dists此时已经是真实世界的距离
    var alpha = addNoise(d.get(":,:-1,:")).getNDArrayInternal.relu().neg().mul(dists)
    val T = alpha.cumSum(1).exp()
    //最前面是1
    alpha = alpha.exp().sub(1).neg()
    val weight = alpha.get(":,:1,:").getNDArrayInternal.concat(new NDList(T.get(":,:-1,:").mul(alpha.get(":,1:,:")), T.get(":,-1:,:")), 1)
    weight
    //输出：
    //weight：尺寸(batchNum, NSamples ( + NImportance), 1)，使用addBkgd(weight2.mul(rgb.getNDArrayInternal.sigmoid()).sum(Array(1)), weight)即可得到输出
  }

  val getSample = if (config.linDisp) (tVals: NDArray, near: NDArray, far: NDArray) => NDArrays.div(1, NDArrays.div(1, near).mul(NDArrays.sub(1, tVals)).add(NDArrays.div(1, far).mul(tVals)))
  else (tVals: NDArray, near: NDArray, far: NDArray) => near.mul(tVals.sub(1).neg()).add(far.mul(tVals))

  val givePerturb = if (config.perturb) (zVals: NDArray) => {
    val manager = zVals.getManager
    val mids = zVals.get(":,1:,:").add(zVals.get(":,:-1,:")).mul(.5)
    val upper = mids.concat(zVals.get(":,-1:,:"), 1)
    val lower = zVals.get(":,:1,:").concat(mids, 1)
    val tRand = manager.randomUniform(0, 1, zVals.getShape)
    lower.add(upper.sub(lower).mul(tRand))
  } else (zVals: NDArray) => zVals

  def getInput(raysO: NDArray, raysD: NDArray, near: NDArray, far: NDArray): (NDArray, NDArray) = {
    //为网络准备输入
    //raysO：尺寸(batchNum,1,3)
    //raysD：尺寸(batchNum,1,3)
    //near：尺寸(batchNum,1)
    //far：尺寸(batchNum,1)
    val manager = raysO.getManager
    val tVals = manager.linspace(0, 1, config.NSamples)
    val zVals = givePerturb(getSample(tVals, near, far).expandDims(2))
    (raysO.add(raysD.mul(zVals)), zVals)
    //输出分别为：
    //pos：尺寸(batchNum, NSamples, 3)
    //zVals：尺寸(batchNum, NSamples, 1)
  }

  val givePerturbPDF = if (config.perturb) (cdf: NDArray) => cdf.getManager.randomUniform(0, 1, Shape.update(cdf.getShape, cdf.getShape.dimension() - 1, config.NImportance), DataType.FLOAT32).sort(-1)
  else (cdf: NDArray) => cdf.getManager.linspace(0, 1, config.NImportance).broadcast(Shape.update(cdf.getShape, cdf.getShape.dimension() - 1, config.NImportance))

  def samplePdf(weight: NDArray, zVals: NDArray, raysO: NDArray, raysD: NDArray): (NDArray, NDArray) = {
    //weight：尺寸(batchNum, NSamples, 1)
    //zVals：尺寸(batchNum, NSamples, 1)
    //raysO：尺寸(batchNum, 1, 3)
    //raysD：尺寸(batchNum, 1, 3)

    val manager = weight.getManager
    val weightCum = weight.get(":,1:-1,0").add(1e-5).stopGradient().cumSum(1)
    val cdf = weightCum.div(weightCum.get(":,-1:"))

    val bins = zVals.get(":,1:,:").sub(zVals.get(":,:-1,:"))
    val u = givePerturbPDF(cdf)

    var inds = cdf.concat(u, -1).argSort(-1).argSort(-1).get(s":,${config.NSamples - 2}:")
    inds = inds.sub(manager.arange(config.NImportance).broadcast(inds.getShape))
    val below = inds.sub(1).maximum(0)
    val above = inds.minimum(config.NSamples - 3)

    val cdfG0 = cdf.get(new NDIndex().addAllDim().addPickDim(below))
    val cdfG1 = cdf.get(new NDIndex().addAllDim().addPickDim(above))
    val binsG0 = bins.get(new NDIndex().addAllDim().addPickDim(below))
    val binsG1 = bins.get(new NDIndex().addAllDim().addPickDim(above))
    var denom = cdfG1.sub(cdfG0)
    denom = NDArrays.where(denom.lt(1e-5), denom.onesLike(), denom)
    val t = u.sub(cdfG0).div(denom)
    val samples = binsG0.add(t.mul(binsG1.sub(binsG0))).expandDims(-1).concat(zVals, -2).sort(-2)

    (raysO.add(raysD.mul(samples)), samples)
    //pos：尺寸(batchNum, NSamples + NImportance, 3)
    //samples：尺寸(batchNum, NSamples + NImportance, 1)
  }
}