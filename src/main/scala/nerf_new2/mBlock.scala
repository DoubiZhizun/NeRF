package nerf_new2

import ai.djl.ndarray._
import ai.djl.ndarray.types._

class mBlock(config: nerfConfig, manager: NDManager) extends nnBlock {

  val posSize = 3 + config.pos_L * 6
  val dirSize = 3 + config.dir_L * 6

  val block = coreBlockGenerator.getMlpBlock(Mf = config.Mf)
  block.initialize(manager, DataType.FLOAT32, new Shape(1, 1, 1, posSize), new Shape(1, 1, dirSize))

  override def forward(input: NDList, training: Boolean): NDList = block.forward(config.ps, input, training)
}