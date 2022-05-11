package dNerf

import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.{NDArray, NDArrays, NDList, NDManager}
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.nn.AbstractBlock
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import dNerf.dNerfBlock._

class dNerfBlock(config: dNerfConfig) extends AbstractBlock(VERSION) {

  private val coarseBlock = if (config.useHierarchical) new dNerfCoreBlock(config, true) else null
  private val fineBlock = new dNerfCoreBlock(config, false)

  if (config.useHierarchical) {
    addChildBlock(coarseBlock.getClass.getSimpleName, coarseBlock)
  }
  addChildBlock(fineBlock.getClass.getSimpleName, fineBlock)

  private val addNoise = if (config.rawNoiseStd > 0) (input: NDArray, training: Boolean) => if (training) input.add(input.getManager.randomNormal(input.getShape).mul(config.rawNoiseStd)) else input
  else (input: NDArray, training: Boolean) => input

  private val addBkgd = if (config.whiteBkgd) (rgb: NDArray, weight: NDArray) => {
    rgb.add(weight.sum(Array(0)).sub(1).neg())
    //对于这个函数的理解：
    //假设渲染出的所有点的颜色都是白色，计算最终渲染出的颜色，
    //然后用纯白色去减这个颜色，就能得到背景所需的颜色
  } else (rgb: NDArray, weight: NDArray) => rgb

  private def getWeight(d: NDArray, zVals: NDArray, training: Boolean): NDArray = {
    //将网络输出的密度计算成权重
    //d：尺寸(NSamples ( + NImportance) - 1, batchNum, 1)
    //zVals：尺寸(NSamples ( + NImportance), batchNum, 1)
    val dists = zVals.get("1:").sub(zVals.get(":-1"))
    //raysD归一化过，此时dists已经是真实世界的距离了
    var alpha = addNoise(d, training).getNDArrayInternal.relu().neg().mul(dists)
    val T = alpha.cumSum(0).exp()
    //最前面是1
    alpha = alpha.exp().sub(1).neg()
    val weight = alpha.get(":1").getNDArrayInternal.concat(new NDList(T.get(":-1").mul(alpha.get("1:")), T.get("-1:")), 0)
    weight
    //输出：
    //weight：尺寸(NSamples ( + NImportance), batchNum, 1)，使用addBkgd(weight.mul(rgb.getNDArrayInternal.sigmoid()).sum(Array(0)), weight)即可得到输出
  }

  private val getSample = if (config.linDisp) (tVals: NDArray, near: NDArray, far: NDArray) => NDArrays.div(1, NDArrays.div(1, near).mul(tVals.sub(1).neg()).add(NDArrays.div(1, far).mul(tVals)))
  else (tVals: NDArray, near: NDArray, far: NDArray) => near.mul(tVals.sub(1).neg()).add(far.mul(tVals))

  private val givePerturb = if (config.perturb) (zVals: NDArray) => {
    val manager = zVals.getManager
    val mids = zVals.get("1:").add(zVals.get(":-1")).mul(.5)
    val upper = mids.concat(zVals.get("-1:"), 0)
    val lower = zVals.get(":1").concat(mids, 0)
    val tRand = manager.randomUniform(0, 1, zVals.getShape)
    lower.add(upper.sub(lower).mul(tRand))
  } else (zVals: NDArray) => zVals

  private def getInput(raysO: NDArray, raysD: NDArray, bounds: NDArray): (NDArray, NDArray) = {
    //为网络准备输入
    //输入在forwardWithCoarse中有介绍
    val manager = raysO.getManager
    val tVals = manager.linspace(0, 1, config.NSamples).expandDims(-1)
    val zVals = givePerturb(getSample(tVals, bounds.get("...,0"), bounds.get("...,1"))).expandDims(-1)
    (raysO.add(raysD.mul(zVals)), zVals)
    //输出分别为：
    //post：尺寸(NSamples, batchNum, 3(4))
    //zVals：尺寸(NSamples, batchNum, 1)
  }

  private val givePerturbPDF = if (config.perturb) (cdf: NDArray) => cdf.getManager.randomUniform(0, 1, Shape.update(cdf.getShape, 0, config.NImportance), DataType.FLOAT32).sort(0)
  else (cdf: NDArray) => cdf.getManager.linspace(0, 1, config.NImportance).repeat(0, cdf.getShape.get(1)).reshape(Shape.update(cdf.getShape, 0, config.NImportance))

  private def samplePdf(weight: NDArray, zVals: NDArray, raysO: NDArray, raysD: NDArray): (NDArray, NDArray) = {
    //weight：尺寸(NSamples, batchNum, 1)
    //zVals：尺寸(NSamples, batchNum, 1)
    //raysO：尺寸(batchNum, 3(4))
    //raysD：尺寸(batchNum, 3(4))

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
    //post：尺寸(batchNum, NSamples + NImportance, 3(4))
    //samples：尺寸(batchNum, NSamples + NImportance, 1)
  }

  private def forwardWithCoarse(parameterStore: ParameterStore, inputs: NDList, training: Boolean): NDList = {
    val (coarsePos, coarseZVals) = getInput(inputs.get(0), inputs.get(1), inputs.get(2))
    val coarseOutput = coarseBlock.forward(parameterStore, new NDList(coarsePos, inputs.get(3), inputs.get(4)), training, null)
    val coarseWeight = getWeight(coarseOutput.get(0), coarseZVals, training)
    val (finePos, fineZVals) = samplePdf(coarseWeight, coarseZVals, inputs.get(0), inputs.get(1))
    val fineOutput = fineBlock.forward(parameterStore, new NDList(finePos, inputs.get(3), inputs.get(4)), training, null)
    val fineWeight = getWeight(fineOutput.get(0), fineZVals, training)
    val fineRgbOut = addBkgd(fineWeight.mul(fineOutput.get(1).getNDArrayInternal.sigmoid()).sum(Array(0)), fineWeight)
    val coarseRgbOut = if (training) addBkgd(coarseWeight.mul(coarseOutput.get(1).getNDArrayInternal.sigmoid()).sum(Array(0)), coarseWeight) else null
    new NDList(fineRgbOut, coarseRgbOut)
  }

  private def forwardWithoutCoarse(parameterStore: ParameterStore, inputs: NDList, training: Boolean): NDList = {
    val (finePos, fineZVals) = getInput(inputs.get(0), inputs.get(1), inputs.get(2))
    val fineOutput = fineBlock.forward(parameterStore, new NDList(finePos, inputs.get(3), inputs.get(4)), training, null)
    val fineWeight = getWeight(fineOutput.get(0), fineZVals, training)
    val fineRgbOut = addBkgd(fineWeight.mul(fineOutput.get(1).getNDArrayInternal.sigmoid()).sum(Array(0)), fineWeight)
    new NDList(fineRgbOut)
    //no coarse
  }

  private val forwardFunction = if (config.useHierarchical) forwardWithCoarse _ else forwardWithoutCoarse _

  override def initializeChildBlocks(manager: NDManager, dataType: DataType, inputShapes: Shape*): Unit = {
    if (config.useHierarchical) {
      if (config.useDir) {
        coarseBlock.initializeChildBlocks(manager, dataType, new Shape(config.NSamples).addAll(inputShapes(0)), inputShapes(3))
      } else {
        coarseBlock.initializeChildBlocks(manager, dataType, new Shape(config.NSamples).addAll(inputShapes(0)))
      }
    }
    if (config.useDir) {
      fineBlock.initializeChildBlocks(manager, dataType, new Shape(config.NSamples + config.NImportance).addAll(inputShapes(0)), inputShapes(3))
    } else {
      fineBlock.initializeChildBlocks(manager, dataType, new Shape(config.NSamples + config.NImportance).addAll(inputShapes(0)))
    }
  }

  override def forwardInternal(parameterStore: ParameterStore, inputs: NDList, training: Boolean, params: PairList[String, AnyRef]): NDList = {
    //输入内容：
    //raysO：尺寸(batchNum, 3(4))
    //raysD：尺寸(batchNum, 3(4))
    //bounds：尺寸(batchNum, 2)
    //viewDir：尺寸(batchNum, 3)
    //time：尺寸(batchNum, 2)
    //没有的东西在位置填充null
    forwardFunction(parameterStore, inputs, training)
    //输出：
    //fineRgbOut：细腻网络渲染结果，尺寸(batchNum, 3)
    //coarseRgbOut：粗糙网络渲染结果，尺寸(batchNum, 3)
    //非训练模式或不使用层级模式下coarseRgbOut为null
  }

  override def getOutputShapes(inputShapes: Array[Shape]): Array[Shape] = {
    if (config.useHierarchical) {
      Array(Shape.update(inputShapes(0), inputShapes(0).dimension() - 1, 3), Shape.update(inputShapes(0), inputShapes(0).dimension() - 1, 3))
    } else {
      Array(Shape.update(inputShapes(0), inputShapes(0).dimension() - 1, 3))
    }
  }
}

object dNerfBlock {
  private val VERSION: Byte = 0
}