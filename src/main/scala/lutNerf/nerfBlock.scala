package lutNerf

import ai.djl.ndarray.index._
import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.nn.core.Linear
import ai.djl.training._
import ai.djl.util._
import lutNerf.nerfBlock._

import java.util.function._
import scala.collection.JavaConverters._

class nerfBlock(config: nerfConfig) extends AbstractBlock(VERSION) {

  val block = new SequentialBlock()
    .add(
      new ParallelBlock(
        toFunction((t: java.util.List[NDList]) => new NDList(NDArrays.concat(new NDList((0 until config.layer).map(t.get(_).get(0)): _*), -1))),
        (0 until config.layer).map(new nerfTable(config.start, _, config.T, config.featureNum): Block).toList.asJava
      )
    )
    .add(
      new ParallelBlock(
        toFunction((t: java.util.List[NDList]) => new NDList(t.get(0).get(0), t.get(1).get(0))),
        List[Block](
          new SequentialBlock()
            .add(Linear.builder().setUnits(64).optBias(false).build())
            .add(Linear.builder().setUnits(64).optBias(false).build())
            .add(Linear.builder().setUnits(3).optBias(false).build()),
          new SequentialBlock()
            .add(Linear.builder().setUnits(64).optBias(false).build())
            .add(Linear.builder().setUnits(64).optBias(false).build())
            .add(Linear.builder().setUnits(1).optBias(false).build())
        ).asJava
      )
    )

  addChildBlock(block.getClass.getSimpleName, block)

  private def getInput(raysO: NDArray, raysD: NDArray, bounds: NDArray): (NDArray, NDArray) = {
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

  private def getWeight(d: NDArray, zVals: NDArray): NDArray = {
    //将网络输出的密度计算成权重
    //d：尺寸(NSamples, batchNum, 1)
    //zVals：尺寸(NSamples, batchNum, 1)
    val dists = zVals.get("1:").sub(zVals.get(":-1"))
    //raysD归一化过，此时dists已经是真实世界的距离了
    var alpha = d.get(":-1").getNDArrayInternal.relu().neg().mul(dists)
    val T = alpha.cumSum(0).exp()
    //最前面是1
    alpha = alpha.exp().sub(1).neg()
    val weight = alpha.get(":1").getNDArrayInternal.concat(new NDList(T.get(":-1").mul(alpha.get("1:")), T.get("-1:")), 0)
    weight
    //输出：
    //weight：尺寸(NSamples, batchNum, 1)，使用addBkgd(weight.mul(rgb.getNDArrayInternal.sigmoid()).sum(Array(0)), weight)即可得到输出
  }

  override def forwardInternal(parameterStore: ParameterStore, inputs: NDList, training: Boolean, params: PairList[String, AnyRef]): NDList = {
    val (pos, zVals) = getInput(inputs.get(0), inputs.get(1), inputs.get(2))
    null
  }

  override def getOutputShapes(inputShapes: Array[Shape]): Array[Shape] = ???
}

object nerfBlock {
  val VERSION: Byte = 0

  private def toFunction[T1, T2](input: T1 => T2): Function[T1, T2] = new Function[T1, T2] {
    override def apply(t: T1): T2 = input(t)
  }
}