package nerf_new2

import ai.djl.modality.cv.Image
import ai.djl.modality.cv.util.NDImageUtils
import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.ndarray.types.Shape
import ai.djl.nn._
import ai.djl.nn.convolutional.Conv2d
import ai.djl.nn.core._

import java.util.function._
import scala.collection.JavaConverters._

object coreBlockGenerator {

  val alphaIndex = new NDIndex("...,:-1,:")

  def getMlpBlock(D: Int = 8, W: Int = 256, Mf: Int = 128, skips: Array[Int] = Array(4)): Block = {
    var block = new SequentialBlock()
    //输入：含有两个元素的NDList，下标为0的是input_ch，下标为1的是input_ch_views
    for (i <- 0 until D) {
      block.add(Linear.builder().setUnits(W).build()).add(Activation.reluBlock())
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
    }).add(Linear.builder().setUnits(W / 2).build())

    val input_viewdirs = new SequentialBlock().add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
      override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow().add(x.get(1).singletonOrThrow().expandDims(-2)))
    }, List[Block](bottleneck, viewdirs).asJava)).add(Activation.reluBlock()).add(Linear.builder().setUnits(Mf).build())
    new SequentialBlock().add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
      override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow(), x.get(1).singletonOrThrow())
    }, List[Block](block, new LambdaBlock(new Function[NDList, NDList] {
      override def apply(x: NDList): NDList = new NDList(x.get(1))
    })).asJava)).add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
      override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow(), x.get(1).singletonOrThrow())
    }, List[Block](input_viewdirs, alpha_out).asJava))
    //输出：NDList尺寸为2，其内容分别为feature和density
  }

  def getCnnBlock(Mf: Int, factor: Int): Block = {
    val block = new SequentialBlock().add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
      override def apply(input: java.util.List[NDList]): NDList = new NDList(input.get(0).singletonOrThrow(), input.get(1).singletonOrThrow())
    }, List[Block](new LambdaBlock(new Function[NDList, NDList] {
      override def apply(input: NDList): NDList = input
    }), Conv2d.builder().setFilters(3).setKernelShape(new Shape(3, 3)).optPadding(new Shape(1, 1)).build()).asJava))
    for (i <- 0 until factor) {
      block.add(new LambdaBlock(new Function[NDList, NDList] {
        override def apply(input: NDList): NDList = new NDList(input.get(0).repeat(Array(1l, 1,2, 2)), input.get(1).repeat(Array(1l,1, 2, 2)))
      })).add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
        override def apply(input: java.util.List[NDList]): NDList = new NDList(input.get(0).get(0), input.get(0).get(1).add(input.get(1).get(1)))
      }, List[Block](new SequentialBlock().add(new Function[NDList, NDList] {
        override def apply(input: NDList): NDList = new NDList(input.get(0))
      }).add(Conv2d.builder().setFilters(Mf >> (i + 1)).setKernelShape(new Shape(3, 3)).optPadding(new Shape(1, 1)).build()).add(Activation.reluBlock()).add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
        override def apply(input: java.util.List[NDList]): NDList = new NDList(input.get(0).singletonOrThrow(), input.get(1).singletonOrThrow())
      }, List[Block](new LambdaBlock(new Function[NDList, NDList] {
        override def apply(input: NDList): NDList = input
      }), Conv2d.builder().setFilters(3).setKernelShape(new Shape(3, 3)).optPadding(new Shape(1, 1)).build()).asJava)), new LambdaBlock(new Function[NDList, NDList] {
        override def apply(input: NDList): NDList = input
      })).asJava))
    }
    block.add(new LambdaBlock(new Function[NDList, NDList] {
      override def apply(input: NDList): NDList = new NDList(input.get(1).getNDArrayInternal.sigmoid())
    }))
  }
}