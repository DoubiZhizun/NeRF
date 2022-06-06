package hashGridNerf

import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.training._
import ai.djl.training.initializer._
import ai.djl.util._
import hashGridNerf.hashGridTable._

class hashGridTable(nLevels: Int, nFeaturesPerLevel: Int, log2HashMapSize: Int, baseResolution: Int) extends AbstractBlock(VERSION) {
  //nLevels：总层数
  //nFeaturesPerLevel：每层的特征数
  //log2HashMapSize：log2(哈希表尺寸）
  //baseResolution：基础分辨率

  val parameters = new Array[Parameter](nLevels) //每一层的参数
  val isHash = new Array[Boolean](nLevels) //每一层是否为哈希表
  val resolution = new Array[Int](nLevels) //每一层的分辨率

  //每一层的分辨率翻一倍

  for (i <- 0 until nLevels) {
    resolution(i) = baseResolution << i
    parameters(i) = addParameter(
      Parameter
        .builder()
        .setName("table")
        .setType(Parameter.Type.OTHER)
        .build()
    )
    parameters(i).setInitializer(new XavierInitializer(XavierInitializer.RandomType.GAUSSIAN, XavierInitializer.FactorType.IN, 2))
  }

  override def prepare(inputShapes: Array[Shape]): Unit = {
    for (i <- 0 until nLevels) {
      val resolutionInLevel = BigInt(baseResolution) << i
      val gridSizeInLevel = resolutionInLevel * resolutionInLevel * resolutionInLevel
      isHash(i) = gridSizeInLevel > (BigInt(1) << log2HashMapSize)
      val paramsInLevel = if (isHash(i)) BigInt(1) << log2HashMapSize else gridSizeInLevel
      parameters(i).setShape(new Shape(paramsInLevel.toLong, nFeaturesPerLevel))
    }
  }

  override def forwardInternal(parameterStore: ParameterStore, inputs: NDList, training: Boolean, params: PairList[String, AnyRef]): NDList = {
    val input = uniform(inputs.get(0))
    val output = new NDList(nLevels)
    for (i <- 0 until nLevels) {
      val inputScaled = input.mul(resolution(i) - 1)
      val inputIndex = (if (isHash(i)) hash(inputScaled.round().toType(DataType.INT32, false))
      else nonHash(inputScaled.round().toType(DataType.INT32, false), resolution(i)))
        .repeat(-1, nFeaturesPerLevel)
      val index = inputIndex.reshape(-1, nFeaturesPerLevel)

      val parameter = parameterStore.getValue(parameters(i), input.getDevice, training)
      output.add(parameter.get(new NDIndex().addPickDim(index)).reshape(inputIndex.getShape))
    }
    new NDList(NDArrays.concat(output, -1))
  }

  override def getOutputShapes(inputShapes: Array[Shape]): Array[Shape] = {
    Array(Shape.update(inputShapes(0), inputShapes(0).dimension() - 1, nLevels * nFeaturesPerLevel))
  }
}

object hashGridTable {
  val VERSION: Byte = 0

  val hashParameters = Array(1, 2654435761L.toInt, 805459861) //哈希参数

  def hash(input: NDArray): NDArray = {
    //哈希函数
    val inputXArray = input.get("...,0").toIntArray
    val inputYArray = input.get("...,1").toIntArray
    val inputZArray = input.get("...,2").toIntArray
    val outputArray = new Array[Int](inputXArray.length)
    for (i <- inputXArray.indices) {
      outputArray(i) = (inputXArray(i) * hashParameters(0)) ^ (inputYArray(i) * hashParameters(1)) ^ (inputZArray(i) * hashParameters(2))
    }
    input.getManager.create(outputArray, Shape.update(input.getShape, input.getShape.dimension() - 1, 1))
  }

  def nonHash(input: NDArray, resolution: Int): NDArray = {
    //非哈希函数
    input.get("...,:1").add(input.get("...,1:2").mul(resolution)).add(input.get("...,2:").mul(resolution * resolution))
  }

  //  def hashIndex(input: NDArray): NDList = {
  //    //哈希函数
  //    val inputXArray = input.get("...,0").toIntArray
  //    val inputYArray = input.get("...,1").toIntArray
  //    val inputZArray = input.get("...,2").toIntArray
  //    val outputArray = Array.fill(8)(new Array[Int](inputXArray.length))
  //    for (i <- inputXArray.indices) {
  //      outputArray(0)(i) = (inputXArray(i) * hashParameters(0)) ^ (inputYArray(i) * hashParameters(1)) ^ (inputZArray(i) * hashParameters(2))
  //      outputArray(1)(i) = ((inputXArray(i) + 1) * hashParameters(0)) ^ (inputYArray(i) * hashParameters(1)) ^ (inputZArray(i) * hashParameters(2))
  //      outputArray(2)(i) = (inputXArray(i) * hashParameters(0)) ^ ((inputYArray(i) + 1) * hashParameters(1)) ^ (inputZArray(i) * hashParameters(2))
  //      outputArray(3)(i) = ((inputXArray(i) + 1) * hashParameters(0)) ^ ((inputYArray(i) + 1) * hashParameters(1)) ^ (inputZArray(i) * hashParameters(2))
  //      outputArray(4)(i) = (inputXArray(i) * hashParameters(0)) ^ (inputYArray(i) * hashParameters(1)) ^ ((inputZArray(i) + 1) * hashParameters(2))
  //      outputArray(5)(i) = ((inputXArray(i) + 1) * hashParameters(0)) ^ (inputYArray(i) * hashParameters(1)) ^ ((inputZArray(i) + 1) * hashParameters(2))
  //      outputArray(6)(i) = (inputXArray(i) * hashParameters(0)) ^ ((inputYArray(i) + 1) * hashParameters(1)) ^ ((inputZArray(i) + 1) * hashParameters(2))
  //      outputArray(7)(i) = ((inputXArray(i) + 1) * hashParameters(0)) ^ ((inputYArray(i) + 1) * hashParameters(1)) ^ ((inputZArray(i) + 1) * hashParameters(2))
  //    }
  //    new NDList(outputArray.map(input.getManager.create(_, Shape.update(input.getShape, input.getShape.dimension() - 1, 1))): _*)
  //  }
  //
  //  def notHashIndex(input: NDArray, resolution: Int): NDList = {
  //    val inputXArray = input.get("...,0").toIntArray
  //    val inputYArray = input.get("...,1").toIntArray
  //    val inputZArray = input.get("...,2").toIntArray
  //    val outputArray = Array.fill(8)(new Array[Int](inputXArray.length))
  //    for (i <- inputXArray.indices) {
  //      outputArray(0)(i) = inputXArray(i) + resolution * inputYArray(i) + resolution * resolution * inputZArray(i)
  //      outputArray(1)(i) = outputArray(0)(i) + 1
  //      outputArray(2)(i) = outputArray(0)(i) + resolution
  //      outputArray(3)(i) = outputArray(1)(i) + resolution
  //      outputArray(4)(i) = outputArray(0)(i) + resolution * resolution
  //      outputArray(5)(i) = outputArray(1)(i) + resolution * resolution
  //      outputArray(6)(i) = outputArray(2)(i) + resolution * resolution
  //      outputArray(7)(i) = outputArray(3)(i) + resolution * resolution
  //    }
  //    new NDList(outputArray.map(input.getManager.create(_, Shape.update(input.getShape, input.getShape.dimension() - 1, 1))): _*)
  //  }

  def uniform(input: NDArray): NDArray = {
    //归一化，小于0的部分用0取代，大于1的部分用1取代
    input.getNDArrayInternal.where(input.lt(0), input.zerosLike()).getNDArrayInternal.where(input.gt(1), input.onesLike())
  }
}