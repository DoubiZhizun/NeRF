package nerf_change

import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.nn._
import ai.djl.nn.core._

import java.util.function._
import scala.collection.JavaConverters._

object coreBlockGenerator {

  val alphaIndex = new NDIndex(":,:-1,:")

  def getBlock(D: Int = 8, W: Int = 256, output_ch: Int = 4, skips: Array[Int] = Array(4)): Block = {
    var block = new SequentialBlock()
    //输入：含有两个元素的NDList，下标为0的是input_ch，下标为1的是input_ch_views
    for (i <- 0 until D) {
      block.add(Linear.builder().setUnits(W).build()).add(new LambdaBlock(new Function[NDList, NDList] {
        override def apply(input: NDList): NDList = new NDList(input.singletonOrThrow().sin())
      }))
      if (skips.contains(i)) {
        val block2 = new LambdaBlock(new Function[NDList, NDList] {
          override def apply(x: NDList): NDList = new NDList(x.singletonOrThrow())
        })
        block = new SequentialBlock().add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
          override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow().concat(x.get(1).singletonOrThrow(), -1))
        }, List[Block](block, block2).asJava))
      }
    }
    block = new SequentialBlock().add(new Function[NDList, NDList] {
      override def apply(x: NDList): NDList = new NDList(x.get(0))
    }).add(block)
    val alpha_out = new SequentialBlock().add(new Function[NDList, NDList] {
      override def apply(x: NDList): NDList = new NDList(x.get(0).get(alphaIndex))
    }).add(Linear.builder().setUnits(1).build())
    val bottleneck = new SequentialBlock().add(new Function[NDList, NDList] {
      override def apply(x: NDList): NDList = new NDList(x.get(0))
    }).add(Linear.builder().setUnits(W / 2).build())
    val viewdirs = new SequentialBlock().add(new Function[NDList, NDList] {
      override def apply(x: NDList): NDList = new NDList(x.get(1))
    }).add(Linear.builder().setUnits(W / 2).build()).add(new Function[NDList, NDList] {
      override def apply(x: NDList): NDList = new NDList(x.singletonOrThrow().expandDims(1))
    })

    val input_viewdirs = new SequentialBlock().add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
      override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow().add(x.get(1).singletonOrThrow()))
    }, List[Block](bottleneck, viewdirs).asJava)).add(new LambdaBlock(new Function[NDList, NDList] {
      override def apply(input: NDList): NDList = new NDList(input.singletonOrThrow().sin())
    })).add(Linear.builder().setUnits(3).build())
    new SequentialBlock().add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
      override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow(), x.get(1).singletonOrThrow())
    }, List[Block](block, new LambdaBlock(new Function[NDList, NDList] {
      override def apply(x: NDList): NDList = new NDList(x.get(1))
    })).asJava)).add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
      override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow(), x.get(1).singletonOrThrow())
    }, List[Block](input_viewdirs, alpha_out).asJava))
    //输出：NDList尺寸为2，其内容分别为RGB和density
  }
}