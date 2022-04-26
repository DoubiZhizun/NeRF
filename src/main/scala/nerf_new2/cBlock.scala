package nerf_new2

import ai.djl.ndarray._
import ai.djl.ndarray.types._

class cBlock(config: nerfConfig, manager: NDManager) extends nnBlock {

  val block = coreBlockGenerator.getCnnBlock(config.Mf, config.factor)
  block.initialize(manager, DataType.FLOAT32, new Shape(1, config.Mf, 8, 8))

  override def forward(input: NDList, training: Boolean): NDList = block.forward(config.ps, input, training)
}