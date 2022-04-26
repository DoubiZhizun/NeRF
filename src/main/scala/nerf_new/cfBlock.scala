package nerf_new

import ai.djl.engine.Engine
import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.training._
import ai.djl.training.initializer._

class cfBlock(config: nerfConfig, manager: NDManager, ps: ParameterStore) extends nnBlock {

  val inputSize = 3 + config.pos_L * 6

  val parameters = new Array[Parameter](27)

  parameters(0) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(256, inputSize)).build()
  parameters(1) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(256)).build()
  parameters(2) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(256, 256)).build()
  parameters(3) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(256)).build()
  parameters(4) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(256, 256)).build()
  parameters(5) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(256)).build()
  parameters(6) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(256, 256)).build()
  parameters(7) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(256)).build()
  parameters(8) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(256, 256)).build()
  parameters(9) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(256)).build()
  parameters(10) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(256, 256)).build()
  parameters(11) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(256)).build()
  parameters(12) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(256, 256)).build()
  parameters(13) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(256)).build()
  parameters(14) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(256, 256)).build()
  parameters(15) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(256)).build()
  parameters(16) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(1, 256)).build()
  parameters(17) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(1)).build()
  parameters(18) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(3, 256)).build()
  parameters(19) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(3)).build()
  parameters(20) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(8, 256)).build()
  parameters(21) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(8)).build()
  parameters(22) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(8, 256)).build()
  parameters(23) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(8)).build()
  parameters(24) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(8, 256)).build()
  parameters(25) = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(8)).build()
  parameters(26) = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(256, inputSize)).build()

  for (p <- parameters) {
    p.initialize(manager, DataType.FLOAT32)
  }

  def activate(input: NDArray): NDArray = input.getNDArrayInternal.relu()

  override def forward(input: NDArray, training: Boolean): (NDArray, NDArray, NDArray) = {
    var temp = input
    for (i <- 0 until 4) {
      val weight = ps.getValue(parameters(2 * i), config.device, training)
      val bias = ps.getValue(parameters(2 * i + 1), config.device, training)
      temp = activate(temp.getNDArrayInternal.linear(temp, weight, bias).get(0))
    }
    var weight = ps.getValue(parameters(8), config.device, training)
    var bias = ps.getValue(parameters(9), config.device, training)
    val weight2 = ps.getValue(parameters(26), config.device, training)
    temp = activate(temp.getNDArrayInternal.linear(temp, weight, bias).get(0).add(input.getNDArrayInternal.linear(input, weight2, null).get(0)))
    for (i <- 5 until 8) {
      val weight = ps.getValue(parameters(2 * i), config.device, training)
      val bias = ps.getValue(parameters(2 * i + 1), config.device, training)
      temp = activate(temp.getNDArrayInternal.linear(temp, weight, bias).get(0))
    }
    weight = ps.getValue(parameters(16), config.device, training)
    bias = ps.getValue(parameters(17), config.device, training)
    val d = temp.getNDArrayInternal.linear(temp.get(":,:-1,:"), weight, bias).get(0)
    weight = ps.getValue(parameters(18), config.device, training)
    bias = ps.getValue(parameters(19), config.device, training)
    val rgb1 = temp.getNDArrayInternal.linear(temp, weight, bias).get(0)
    weight = ps.getValue(parameters(20), config.device, training)
    bias = ps.getValue(parameters(21), config.device, training)
    val rgb2 = temp.getNDArrayInternal.linear(temp, weight, bias).get(0)
    weight = ps.getValue(parameters(22), config.device, training)
    bias = ps.getValue(parameters(23), config.device, training)
    val rgb3 = temp.getNDArrayInternal.linear(temp, weight, bias).get(0)
    weight = ps.getValue(parameters(24), config.device, training)
    bias = ps.getValue(parameters(25), config.device, training)
    val rgb4 = temp.getNDArrayInternal.linear(temp, weight, bias).get(0)
    (rgb1, rgb2.getNDArrayInternal.stack(new NDList(rgb3, rgb4), -2), d)
  }
}