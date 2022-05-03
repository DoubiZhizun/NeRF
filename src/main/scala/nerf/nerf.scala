package nerf

import ai.djl.engine._
import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.ndarray.types._
import ai.djl.nn.{AbstractBlock, Block, BlockList, Parameter, ParameterList}
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.Initializer
import ai.djl.training.loss._
import ai.djl.util.PairList

import java.io.{DataInputStream, DataOutputStream}
import java.util.function.Predicate

class nerf(config: nerfConfig, manager: NDManager){

  val loss = Loss.l2Loss("L2Loss", 1)
  val coarseBlock = if (config.useHierarchical) new nerfModel(config, true).initialize(manager) else null
  val fineBlock = new nerfModel(config, false).initialize(manager)

  def predict(raysO: NDArray, raysD: NDArray, time: NDArray, bounds: NDArray, viewdir: NDArray): NDArray = {
    noise(false)
    val (_, fineRgbOut) = forward(raysO, raysD, time, bounds, viewdir, false)
    fineRgbOut
  }

  def train(raysO: NDArray, raysD: NDArray, time: NDArray, bounds: NDArray, viewdir: NDArray, label: NDArray): Float = {
    //label：尺寸(batchNum, 3)
    noise(true)
    val collector = Engine.getInstance().newGradientCollector()
    val (coarseRgbOut, fineRgbOut) = forward(raysO, raysD, time, bounds, viewdir, true)
    val lossValue = lossCalculate(coarseRgbOut, fineRgbOut, label)
    collector.backward(lossValue)
    collector.close()
    config.ps.updateAllParameters()
    lossValue.getFloat()
  }

  val forward = if (config.useHierarchical) forwardWithCoarse _ else forwardWithoutCoarse _
  val lossCalculate = if (config.useHierarchical) (coarse: NDArray, fine: NDArray, label: NDArray) => loss.evaluate(new NDList(label), new NDList(coarse)).add(loss.evaluate(new NDList(label), new NDList(fine))) else (coarse: NDArray, fine: NDArray, label: NDArray) => loss.evaluate(new NDList(label), new NDList(fine))

  def forwardWithCoarse(raysO: NDArray, raysD: NDArray, time: NDArray, bounds: NDArray, viewdir: NDArray, training: Boolean): (NDArray, NDArray) = {
    //输入在getInput中有详细介绍
    //raysO：尺寸(batchNum, 3)
    //raysD：尺寸(batchNum, 3)
    //time：尺寸(batchNum, 1)或(1)
    //bounds：尺寸(batchNum, 2)
    //viewdir：尺寸(batchNum, 3)
    val (coarsePos, coarseZVals) = getInput(raysO, raysD, bounds)
    val (coarseD, coarseRgb) = coarseBlock.forward(coarsePos, viewdir, time, training)
    val coarseWeight = getWeight(coarseD, coarseZVals)
    val (finePos, fineZVals) = samplePdf(coarseWeight, coarseZVals, raysO, raysD)
    val (fineD, fineRgb) = fineBlock.forward(finePos, viewdir, time, training)
    val fineWeight = getWeight(fineD, fineZVals)
    val fineRgbOut = addBkgd(fineWeight.mul(fineRgb.getNDArrayInternal.sigmoid()).sum(Array(0)), fineWeight)
    val coarseRgbOut = if (training) addBkgd(coarseWeight.mul(coarseRgb.getNDArrayInternal.sigmoid()).sum(Array(0)), coarseWeight) else null
    (coarseRgbOut, fineRgbOut)
    //输出：
    //coarseRgbOut：粗糙网络渲染结果，尺寸(batchNum, 3)
    //fineRgbOut：细腻网络渲染结果，尺寸(batchNum, 3)
  }

  def forwardWithoutCoarse(raysO: NDArray, raysD: NDArray, time: NDArray, bounds: NDArray, viewdir: NDArray, training: Boolean): (NDArray, NDArray) = {
    val (finePos, fineZVals) = getInput(raysO, raysD, bounds)
    val (findD, fineRgb) = fineBlock.forward(finePos, viewdir, time, training)
    val fineWeight = getWeight(findD, fineZVals)
    val fineRgbOut = addBkgd(fineWeight.mul(fineRgb.getNDArrayInternal.sigmoid()).sum(Array(0)), fineWeight)
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
    rgb.add(weight.sum(Array(0)).sub(1).neg())
    //对于这个函数的理解：
    //假设渲染出的所有点的颜色都是白色，计算最终渲染出的颜色，
    //然后用纯白色去减这个颜色，就能得到背景所需的颜色
  } else (rgb: NDArray, weight: NDArray) => rgb

  def getWeight(d: NDArray, zVals: NDArray): NDArray = {
    //将网络输出的密度计算成权重
    //d：尺寸(NSamples ( + NImportance) - 1, batchNum, 1)
    //zVals：尺寸(NSamples ( + NImportance), batchNum, 1)
    val dists = zVals.get("1:").sub(zVals.get(":-1"))
    //raysD归一化过，dists此时已经是真实世界的距离
    var alpha = addNoise(d).getNDArrayInternal.relu().neg().mul(dists)
    val T = alpha.cumSum(0).exp()
    //最前面是1
    alpha = alpha.exp().sub(1).neg()
    val weight = alpha.get(":1").getNDArrayInternal.concat(new NDList(T.get(":-1").mul(alpha.get("1:")), T.get("-1:")), 0)
    weight
    //输出：
    //weight：尺寸(NSamples ( + NImportance), batchNum, 1)，使用addBkgd(weight.mul(rgb.getNDArrayInternal.sigmoid()).sum(Array(0)), weight)即可得到输出
  }

  val getSample = if (config.linDisp) (tVals: NDArray, near: NDArray, far: NDArray) => NDArrays.div(1, NDArrays.div(1, near).mul(tVals.sub(1).neg()).add(NDArrays.div(1, far).mul(tVals)))
  else (tVals: NDArray, near: NDArray, far: NDArray) => near.mul(tVals.sub(1).neg()).add(far.mul(tVals))

  val givePerturb = if (config.perturb) (zVals: NDArray) => {
    val manager = zVals.getManager
    val mids = zVals.get("1:").add(zVals.get(":-1")).mul(.5)
    val upper = mids.concat(zVals.get("-1:"), 0)
    val lower = zVals.get(":1").concat(mids, 0)
    val tRand = manager.randomUniform(0, 1, zVals.getShape)
    lower.add(upper.sub(lower).mul(tRand))
  } else (zVals: NDArray) => zVals

  def getInput(raysO: NDArray, raysD: NDArray, bounds: NDArray): (NDArray, NDArray) = {
    //为网络准备输入
    //输入在forwardWithCoarse中有介绍
    val manager = raysO.getManager
    val tVals = manager.linspace(0, 1, config.NSamples).expandDims(-1)
    val zVals = givePerturb(getSample(tVals, bounds.get("...,0"), bounds.get("...,1"))).expandDims(-1)
    (raysO.add(raysD.mul(zVals)), zVals)
    //输出分别为：
    //pos：尺寸(NSamples, batchNum, 3)
    //zVals：尺寸(NSamples, batchNum, 1)
  }

  val givePerturbPDF = if (config.perturb) (cdf: NDArray) => cdf.getManager.randomUniform(0, 1, Shape.update(cdf.getShape, 0, config.NImportance), DataType.FLOAT32).sort(0)
  else (cdf: NDArray) => cdf.getManager.linspace(0, 1, config.NImportance).repeat(0, cdf.getShape.get(1)).reshape(Shape.update(cdf.getShape, 0, config.NImportance))

  def samplePdf(weight: NDArray, zVals: NDArray, raysO: NDArray, raysD: NDArray): (NDArray, NDArray) = {
    //weight：尺寸(NSamples, batchNum, 1)
    //zVals：尺寸(NSamples, batchNum, 1)
    //raysO：尺寸(batchNum, 3)
    //raysD：尺寸(batchNum, 3)

    val manager = weight.getManager
    val weightCum = weight.stopGradient().get("1:-1").add(1e-5).cumSum(0)
    val sum = weightCum.get("-1:")
    val cdf = sum.zerosLike().concat(weightCum.div(sum), 0)
    //大小NSamples - 1

    val bins = zVals.get("1:").add(zVals.get(":-1")).mul(.5)
    val u = givePerturbPDF(cdf)

    var inds = cdf.concat(u, 0).argSort(0).argSort(0).get(s"${cdf.getShape.get(0)}:")
    inds = inds.sub(manager.arange(config.NImportance).expandDims(-1).expandDims(-1))
    val below = inds.sub(1).maximum(0)
    val above = inds.minimum(cdf.getShape.get(0) - 1)

    val cdfG0 = cdf.get(new NDIndex().addPickDim(below))
    val cdfG1 = cdf.get(new NDIndex().addPickDim(above))
    val binsG0 = bins.get(new NDIndex().addPickDim(below))
    val binsG1 = bins.get(new NDIndex().addPickDim(above))
    var denom = cdfG1.sub(cdfG0)
    denom = NDArrays.where(denom.lt(1e-5), denom.onesLike(), denom)
    val t = u.sub(cdfG0).div(denom)
    val samples = binsG0.add(t.mul(binsG1.sub(binsG0))).concat(zVals, 0).sort(0)

    (raysO.add(raysD.mul(samples)), samples)
    //pos：尺寸(batchNum, NSamples + NImportance, 3)
    //samples：尺寸(batchNum, NSamples + NImportance, 1)
  }

  def save(os: DataOutputStream): Unit = {
    if (config.NImportance > 0) {
      coarseBlock.save(os)
    }
    fineBlock.save(os)
  }

  def load(manager: NDManager, is: DataInputStream): Unit = {
    if (config.NImportance > 0) {
      coarseBlock.load(manager, is)
    }
    fineBlock.load(manager, is)
  }
}