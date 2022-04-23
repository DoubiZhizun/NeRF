package nerf_new

import ai.djl.ndarray._

abstract class nnBlock {

  //神经网络模块抽象类

  def setRequireGradient(input: Boolean): Unit

  def forward(input: NDArray): (NDArray, NDArray, NDArray)

  def updateParameters(lr: Double): Unit

}