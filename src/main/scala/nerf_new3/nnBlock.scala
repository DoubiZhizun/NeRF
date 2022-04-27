package nerf_new3

import ai.djl.ndarray._

abstract class nnBlock {

  //神经网络模块抽象类

  def forward(input: NDList, training: Boolean): NDList

}