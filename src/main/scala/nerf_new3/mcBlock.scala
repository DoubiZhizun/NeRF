package nerf_new3

import ai.djl.ndarray._
import ai.djl.ndarray.types._

class mcBlock(config: nerfConfig, manager: NDManager) extends nnBlock {

  val block = coreBlockGenerator.getBlock(Mf = config.Mf, factor = config.factor)
  block.initialize(manager, DataType.FLOAT32, new Shape(1, 1, 1, 3 + config.pos_L * 6))

  override def forward(input: NDList, training: Boolean): NDList = block.forward(config.ps, input, training)
}