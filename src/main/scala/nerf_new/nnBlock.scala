package nerf_new

import ai.djl.ndarray._

abstract class nnBlock {

  //神经网络模块抽象类

  def forward(input: NDArray, training: Boolean): (NDArray, NDArray, NDArray)

}