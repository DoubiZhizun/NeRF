package nerf_origin

import ai.djl._
import ai.djl.ndarray._
import ai.djl.nn._
import ai.djl.nn.core._

import java.util.function._
import scala.collection.JavaConverters._
import scala.collection.mutable._

object run_nerf_helpers {

  case class Embedder(include_input: Boolean, input_dims: Int, max_freq_log2: Int, num_freqs: Int, log_sampling: Boolean, periodic_fns: Array[NDArray => NDArray]) {
    val embed_fns = new ArrayBuffer[NDArray => NDArray]
    var out_dim = 0
    if (include_input) {
      embed_fns += (x => x)
      out_dim += input_dims
    }
    val manager = NDManager.newBaseManager()
    val freq_bands = if (log_sampling) manager.linspace(0, max_freq_log2, num_freqs).toFloatArray.map(f => Math.pow(2, f).toFloat) else manager.linspace(1, 1 << max_freq_log2, num_freqs).toFloatArray
    for (freq <- freq_bands) {
      for (p_fn <- periodic_fns) {
        embed_fns += (x => p_fn(x.mul(freq)))
        out_dim += input_dims
      }
    }
    manager.close()

    def embeded(inputs: NDArray): NDArray = {
      if (embed_fns.isEmpty) {
        null
      } else {
        NDArrays.concat(new NDList(embed_fns.map(fn => fn(inputs)): _*))
      }
    }
  }

  def get_embedder(multires: Int, i: Int = 0): (NDArray => NDArray, Int) = {
    if (i == -1) {
      return ((x: NDArray) => x.get(), 3)
    }
    val embedder_obj = Embedder(true, 3, multires - 1, multires, true, Array(x => x.sin(), x => x.cos()))
    (embedder_obj.embeded(_), embedder_obj.out_dim)
  }

  def init_nerf_model(D: Int = 8, W: Int = 256, output_ch: Int = 4, skips: Array[Int] = Array(4), use_viewdirs: Boolean = false): Block = {
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
    if (use_viewdirs) {
      val alpha_out = new SequentialBlock().add(new Function[NDList, NDList] {
        override def apply(x: NDList): NDList = new NDList(x.get(0))
      }).add(Linear.builder().setUnits(1).build())
      val bottleneck = new SequentialBlock().add(new Function[NDList, NDList] {
        override def apply(x: NDList): NDList = new NDList(x.get(0))
      }).add(Linear.builder().setUnits(256).build())
      val block2 = new LambdaBlock(new Function[NDList, NDList] {
        override def apply(x: NDList): NDList = new NDList(x.get(1))
      })
      val input_viewdirs = new SequentialBlock().add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
        override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow().concat(x.get(1).singletonOrThrow(), -1))
      }, List[Block](bottleneck, block2).asJava)).add(Linear.builder().setUnits(W / 2).build()).add(Activation.reluBlock())
        .add(Linear.builder().setUnits(3).build())
      new SequentialBlock().add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
        override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow(), x.get(1).singletonOrThrow())
      }, List[Block](block, block2).asJava)).add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
        override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow().concat(x.get(1).singletonOrThrow(), -1))
      }, List[Block](input_viewdirs, alpha_out).asJava))
      //输出：NDList尺寸为1，其内容大小为4
    } else {
      block.add(Linear.builder().setUnits(output_ch).build())
      //输出：NDList尺寸为1，其内容大小为output_ch
    }
  }
}