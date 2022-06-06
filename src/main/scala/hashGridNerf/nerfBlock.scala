package hashGridNerf

import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.nn.core.Linear
import ai.djl.training._
import ai.djl.util._
import hashGridNerf.nerfBlock._

import java.util.function._
import scala.collection.JavaConverters._

class nerfBlock(config: nerfConfig) extends AbstractBlock(VERSION) {

  val block = new SequentialBlock()
    .add(
      new ParallelBlock(
        toFunction((t: java.util.List[NDList]) => new NDList(t.get(0).get(0).concat(t.get(1).get(0), -1))),
        List[Block](
          new hashGridTable(config.layer, config.featureNum, config.T, 1 << config.start),
          new LambdaBlock(toFunction((t: NDList) => new NDList(positionCode(t.get(1), config.dirL))))
        ).asJava
      )
    )
    .add(
      new ParallelBlock(
        toFunction((t: java.util.List[NDList]) => new NDList(t.get(0).get(0), t.get(1).get(0))),
        List[Block](
          new SequentialBlock()
            .add(Linear.builder().setUnits(64).build()).add(Activation.reluBlock())
            .add(Linear.builder().setUnits(64).build()).add(Activation.reluBlock())
            .add(Linear.builder().setUnits(3).build()),
          new SequentialBlock()
            .add(Linear.builder().setUnits(64).build()).add(Activation.reluBlock())
            .add(Linear.builder().setUnits(1).build())
        ).asJava
      )
    )

  addChildBlock(block.getClass.getSimpleName, block)

  def getInput(raysO: NDArray, raysD: NDArray, bounds: NDArray): (NDArray, NDArray) = {
    //为网络准备输入
    //输入在forwardWithCoarse中有介绍
    val manager = raysO.getManager
    val tVals = manager.linspace(0, 1, config.NSamples).expandDims(-1)
    val zVals = bounds.get("...,0").mul(tVals.sub(1).neg()).add(bounds.get("...,1").mul(tVals))
    (raysO.add(raysD.mul(zVals)), zVals)
    //输出分别为：
    //post：尺寸(NSamples, batchNum, 3)
    //zVals：尺寸(NSamples, batchNum, 1)
  }

  private val addNoise = if (config.rawNoiseStd > 0) (input: NDArray, training: Boolean) => if (training) input.add(input.getManager.randomNormal(input.getShape).mul(config.rawNoiseStd)) else input
  else (input: NDArray, training: Boolean) => input

  def getWeight(d: NDArray, zVals: NDArray, training: Boolean): NDArray = {
    //将网络输出的密度计算成权重
    //d：尺寸(NSamples, batchNum, 1)
    //zVals：尺寸(NSamples, batchNum, 1)
    val dists = zVals.get("1:").sub(zVals.get(":-1"))
    //raysD归一化过，此时dists已经是真实世界的距离了
    var alpha = addNoise(d.get(":-1"), training).getNDArrayInternal.relu().neg().mul(dists)
    val T = alpha.cumSum(0).exp()
    //最前面是1
    alpha = alpha.exp().sub(1).neg()
    val weight = alpha.get(":1").getNDArrayInternal.concat(new NDList(T.get(":-1").mul(alpha.get("1:")), T.get("-1:")), 0)
    weight
    //输出：
    //weight：尺寸(NSamples, batchNum, 1)，使用addBkgd(weight.mul(rgb.getNDArrayInternal.sigmoid()).sum(Array(0)), weight)即可得到输出
  }

  private val addBkgd = if (config.whiteBkgd) (rgb: NDArray, weight: NDArray) => {
    rgb.add(weight.sum(Array(0)).sub(1).neg())
    //对于这个函数的理解：
    //假设渲染出的所有点的颜色都是白色，计算最终渲染出的颜色，
    //然后用纯白色去减这个颜色，就能得到背景所需的颜色
  } else (rgb: NDArray, weight: NDArray) => rgb

  override def forwardInternal(parameterStore: ParameterStore, inputs: NDList, training: Boolean, params: PairList[String, AnyRef]): NDList = {
    val (pos, zVals) = getInput(inputs.get(0), inputs.get(1), inputs.get(2))
    val blockOutput = block.forward(parameterStore, new NDList(pos, inputs.get(3)), training, params)
    val weight = getWeight(blockOutput.get(1), zVals, training)
    val rgbOut = addBkgd(blockOutput.get(0), weight)
    new NDList(rgbOut)
  }

  override def getOutputShapes(inputShapes: Array[Shape]): Array[Shape] = {
    Array(Shape.update(inputShapes(0), inputShapes(0).dimension() - 1, 3))
  }
}

object nerfBlock {
  val VERSION: Byte = 0

  def toFunction[T1, T2](input: T1 => T2): Function[T1, T2] = new Function[T1, T2] {
    override def apply(t: T1): T2 = input(t)
  }

  def positionCode(input: NDArray, L: Int): NDArray = {
    //sin cos位置编码，L为编码阶数
    val output = new NDList(L * 2)
    for (i <- 0 until L) {
      val inputMulFactor = input.mul(1 << i) //原文貌似就没有PI？
      output.add(inputMulFactor.sin())
      output.add(inputMulFactor.cos())
    }
    input.getNDArrayInternal.concat(output, -1)
  }
}