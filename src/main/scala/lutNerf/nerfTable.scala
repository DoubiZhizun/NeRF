package lutNerf

import ai.djl.ndarray.index._
import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.training._
import ai.djl.training.initializer._
import ai.djl.util._
import lutNerf.nerfTable._

class nerfTable(start: Int, layer: Int, T: Int, featureNum: Int) extends AbstractBlock(VERSION) {

  private val isHash = (start + layer) * 3 > T

  private val table = addParameter(
    Parameter
      .builder()
      .setName("table")
      .setType(Parameter.Type.OTHER)
      .build()
  )
  table.setInitializer(new XavierInitializer(XavierInitializer.RandomType.GAUSSIAN, XavierInitializer.FactorType.IN, 2))

  private val size = 1 << (start + layer)

  private val pi = Array(1, 2654435761L.toInt, 805459861)
  private val mask = (1 << T) - 1
  //哈希函数的参数

  private def hash(input: NDArray): NDList = {
    //哈希函数
    val inputXArray = input.get("...,0").toIntArray
    val inputYArray = input.get("...,1").toIntArray
    val inputZArray = input.get("...,2").toIntArray
    val outputArray = Array.fill(8)(new Array[Int](inputXArray.length))
    for (i <- inputXArray.indices) {
      outputArray(0)(i) = (inputXArray(i) * pi(0)) ^ (inputYArray(i) * pi(1)) ^ (inputZArray(i) * pi(2))
      outputArray(1)(i) = ((inputXArray(i) + 1) * pi(0)) ^ (inputYArray(i) * pi(1)) ^ (inputZArray(i) * pi(2))
      outputArray(2)(i) = (inputXArray(i) * pi(0)) ^ ((inputYArray(i) + 1) * pi(1)) ^ (inputZArray(i) * pi(2))
      outputArray(3)(i) = ((inputXArray(i) + 1) * pi(0)) ^ ((inputYArray(i) + 1) * pi(1)) ^ (inputZArray(i) * pi(2))
      outputArray(4)(i) = (inputXArray(i) * pi(0)) ^ (inputYArray(i) * pi(1)) ^ ((inputZArray(i) + 1) * pi(2))
      outputArray(5)(i) = ((inputXArray(i) + 1) * pi(0)) ^ (inputYArray(i) * pi(1)) ^ ((inputZArray(i) + 1) * pi(2))
      outputArray(6)(i) = (inputXArray(i) * pi(0)) ^ ((inputYArray(i) + 1) * pi(1)) ^ ((inputZArray(i) + 1) * pi(2))
      outputArray(7)(i) = ((inputXArray(i) + 1) * pi(0)) ^ ((inputYArray(i) + 1) * pi(1)) ^ ((inputZArray(i) + 1) * pi(2))
    }
    new NDList(outputArray.map(input.getManager.create(_, Shape.update(input.getShape, input.getShape.dimension() - 1, 1))): _*)
  }

  private def notHash(input: NDArray): NDList = {
    val inputXArray = input.get("...,0").toIntArray
    val inputYArray = input.get("...,1").toIntArray
    val inputZArray = input.get("...,2").toIntArray
    val outputArray = Array.fill(8)(new Array[Int](inputXArray.length))
    for (i <- inputXArray.indices) {
      outputArray(0)(i) = inputXArray(i) + size * inputYArray(i) + size * size * inputZArray(i)
      outputArray(1)(i) = outputArray(0)(i) + 1
      outputArray(2)(i) = outputArray(0)(i) + size
      outputArray(3)(i) = outputArray(1)(i) + size
      outputArray(4)(i) = outputArray(0)(i) + size * size
      outputArray(5)(i) = outputArray(1)(i) + size * size
      outputArray(6)(i) = outputArray(2)(i) + size * size
      outputArray(7)(i) = outputArray(3)(i) + size * size
    }
    new NDList(outputArray.map(input.getManager.create(_, Shape.update(input.getShape, input.getShape.dimension() - 1, 1))): _*)
  }

  private def getInvalid(input: NDArray): NDArray = {
    val lt0 = input.lt(0)
    val gte1 = input.gte(1)
    lt0.get("...,0").logicalOr(lt0.get("...,1")).logicalOr(lt0.get("...,2"))
      .logicalOr(gte1.get("...,0").logicalOr(gte1.get("...,1")).logicalOr(gte1.get("...,2")))
  }

  override def prepare(inputShapes: Array[Shape]): Unit = {
    if (isHash) {
      table.setShape(new Shape(1 << T, featureNum))
    } else {
      table.setShape(new Shape(size * size * size, featureNum))
    }
  }

  override def forwardInternal(parameterStore: ParameterStore, inputs: NDList, training: Boolean, params: PairList[String, AnyRef]): NDList = {
    val input = inputs.get(layer)
    val invalid = getInvalid(input).expandDims(-1) //不符合要求的部分

    val inputScaled = input.getNDArrayInternal.where(invalid, invalid.getManager.zeros(invalid.getShape))
    val inputIndex = inputScaled.floor().toType(DataType.INT32, false)
    val inputW = inputScaled.sub(inputIndex)

    val index = if (isHash) hash(inputIndex) else notHash(inputIndex)
    val tableParameters = parameterStore.getValue(table, input.getDevice, training)

    val tableOutput = new NDList(8)
    for (i <- 0 until 8) {
      tableOutput.add(tableParameters.get(new NDIndex().addPickDim(index.get(i).repeat(-1, featureNum))))
    }

    val tableOutput2 = new NDList(4)
    val inputWx = inputW.get("...,:1")
    for (i <- 0 until 4) {
      tableOutput2.add(tableOutput.get(2 * i + 1).sub(tableOutput.get(2 * i)).mul(inputWx).add(tableOutput.get(2 * i)))
    }

    val tableOutput3 = new NDList(2)
    val inputWy = inputW.get("...,1:2")
    for (i <- 0 until 2) {
      tableOutput3.add(tableOutput2.get(2 * i + 1).sub(tableOutput2.get(2 * i)).mul(inputWy).add(tableOutput2.get(2 * i)))
    }

    val finalOutput = tableOutput3.get(1).sub(tableOutput3.get(0)).mul(inputW.get("...,2:")).add(tableOutput3.get(0))

    new NDList(finalOutput.getNDArrayInternal.where(invalid, invalid.getManager.zeros(invalid.getShape)))
  }

  override def getOutputShapes(inputShapes: Array[Shape]): Array[Shape] = {
    val inputShape = inputShapes(layer)
    Array(Shape.update(inputShape, inputShape.dimension() - 1, featureNum))
  }
}

object nerfTable {
  val VERSION: Byte = 0
}