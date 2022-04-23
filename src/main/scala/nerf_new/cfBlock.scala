package nerf_new

import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.ndarray.types._

class cfBlock(config: nerfConfig, manager: NDManager) extends nnBlock {

  val inputSize = config.pos_L * 6 + 3

  val parameters = new NDList(26)

  parameters.add(manager.randomNormal(new Shape(256, inputSize)))
  parameters.add(manager.randomNormal(new Shape(256)))
  parameters.add(manager.randomNormal(new Shape(256, 256)))
  parameters.add(manager.randomNormal(new Shape(256)))
  parameters.add(manager.randomNormal(new Shape(256, 256)))
  parameters.add(manager.randomNormal(new Shape(256)))
  parameters.add(manager.randomNormal(new Shape(256, 256)))
  parameters.add(manager.randomNormal(new Shape(256)))
  parameters.add(manager.randomNormal(new Shape(256, 256)))
  parameters.add(manager.randomNormal(new Shape(256)))
  parameters.add(manager.randomNormal(new Shape(256, 256)))
  parameters.add(manager.randomNormal(new Shape(256)))
  parameters.add(manager.randomNormal(new Shape(256, 256)))
  parameters.add(manager.randomNormal(new Shape(256)))
  parameters.add(manager.randomNormal(new Shape(256, 256)))
  parameters.add(manager.randomNormal(new Shape(256)))
  parameters.add(manager.randomNormal(new Shape(1, 256)))
  parameters.add(manager.randomNormal(new Shape(1)))
  parameters.add(manager.randomNormal(new Shape(3, 256)))
  parameters.add(manager.randomNormal(new Shape(3)))
  parameters.add(manager.randomNormal(new Shape(8, 256)))
  parameters.add(manager.randomNormal(new Shape(8)))
  parameters.add(manager.randomNormal(new Shape(8, 256)))
  parameters.add(manager.randomNormal(new Shape(8)))
  parameters.add(manager.randomNormal(new Shape(8, 256)))
  parameters.add(manager.randomNormal(new Shape(8)))

  def activate(input: NDArray): NDArray = input.sin()


  override def setRequireGradient(input: Boolean): Unit = {
    for (i <- 0 until parameters.size()) {
      parameters.get(i).setRequiresGradient(input)
    }
  }

  override def forward(input: NDArray): (NDArray, NDArray, NDArray) = {
    var temp = input
    for (i <- 0 until 8) {
      temp = activate(temp.getNDArrayInternal.linear(temp, parameters.get(2 * i), parameters.get(2 * i + 1)).get(0))
    }
    val d = temp.getNDArrayInternal.linear(temp.get(new NDIndex(":,:-1,:")), parameters.get(16), parameters.get(17)).get(0)
    val rgb1 = temp.getNDArrayInternal.linear(temp, parameters.get(18), parameters.get(19)).get(0)
    val rgb2 = temp.getNDArrayInternal.linear(temp, parameters.get(20), parameters.get(21)).get(0)
    val rgb3 = temp.getNDArrayInternal.linear(temp, parameters.get(22), parameters.get(23)).get(0)
    val rgb4 = temp.getNDArrayInternal.linear(temp, parameters.get(24), parameters.get(25)).get(0)
    (rgb1, rgb2.getNDArrayInternal.stack(new NDList(rgb3, rgb4), -2), d)
  }

  override def updateParameters(lr: Double): Unit = {
    for (i <- 0 until parameters.size()) {
      val grad = parameters.get(i).getGradient
      val gradWithLr = grad.mul(lr)
      parameters.get(i).setRequiresGradient(false)
      parameters.get(i).subi(gradWithLr)
      grad.close()
      gradWithLr.close()
    }
  }
}