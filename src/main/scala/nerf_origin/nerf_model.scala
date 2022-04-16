package nerf_origin

import ai.djl._
import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.nn.core._
import ai.djl.training._
import ai.djl.training.dataset._
import ai.djl.training.loss._
import ai.djl.training.optimizer._
import ai.djl.training.tracker._

import java.util.function._
import scala.collection.JavaConverters._
import scala.collection.mutable._

class nerf_model(args: Config, manager: NDManager) {

  //create nerf
  val (embed_fn, input_ch) = get_embedder(args.multires, args.i_embed)
  var input_ch_views = 0
  var embeddirs_fn: NDArray => NDArray = null
  if (args.use_viewdirs) {
    val temp = get_embedder(args.multires_views, args.i_embed)
    input_ch_views = temp._2
    embeddirs_fn = temp._1
  }
  val model = Model.newInstance("coarse")
  model.setBlock(init_nerf_block(D = args.netdepth, W = args.netwidth, output_ch = 4, skips = Array(4), use_viewdirs = args.use_viewdirs))
  var model_fine: Model = null
  if (args.N_importance > 0) {
    model_fine = Model.newInstance("fine")
    model_fine.setBlock(init_nerf_block(D = args.netdepth_fine, W = args.netwidth_fine, output_ch = 4, skips = Array(4), use_viewdirs = args.use_viewdirs))
  }
  val sgd = Optimizer.sgd().setLearningRateTracker(Tracker.factor().setBaseValue(args.lrate.toFloat).setFactor(Math.pow(0.1, 1.0 / (args.lrate_decay * 1000)).toFloat).build()).build()
  val sgd_fine = Optimizer.sgd().setLearningRateTracker(Tracker.factor().setBaseValue(args.lrate.toFloat).setFactor(Math.pow(0.1, 1.0 / (args.lrate_decay * 1000)).toFloat).build()).build()

  val trainer = model.newTrainer(new DefaultTrainingConfig(Loss.l2Loss("L2Loss", 1)).optOptimizer(sgd))
  val trainer_fine = model_fine.newTrainer(new DefaultTrainingConfig(Loss.l2Loss("L2Loss_fine", 1)).optOptimizer(sgd_fine))

  def render_rays(ray_batch: Batch, lindisp: Boolean, N_samples: Int, perturb: Double, train: Boolean = false): Unit = {
    /* ray_batch中的内容：
     * ray_batch.getData中每个数据的第0维度为batch中的不同组数据
     * ray_batch.getData中各个数据的第1维度为：
     * get(0) ray origin
     * get(1) ray direction
     * get(2) near
     * get(3) far
     * get(4) viewing direction
     *
     * lindist：如果为真，采样在逆深度（视差）下线性，否则在深度下线性
     *
     * N_samples：每条光线的采样数量
     *
     * perturb：如果不是0则每条光线都随机采样
     */
    val N_rays = ray_batch.getSize
    val rays_o = ray_batch.getData.get(0)
    val rays_d = ray_batch.getData.get(1)
    val near = ray_batch.getData.get(2).reshape(-1, 1)
    val far = ray_batch.getData.get(3).reshape(-1, 1)
    val viewdirs = if (ray_batch.getData.size() > 3) ray_batch.getData.get(4) else null

    val manager = ray_batch.getManager
    val t_vals = manager.linspace(0, 1, N_samples)
    var z_vals: NDArray = null
    if (!lindisp) {
      //在near和far之间均匀采样
      z_vals = near.mul(NDArrays.sub(1, t_vals)).add(far.mul(t_vals))
    } else {
      z_vals = NDArrays.div(1, NDArrays.div(1, near).mul(NDArrays.sub(1, t_vals)).add(NDArrays.div(1, far).mul(t_vals)))
    }
    z_vals = z_vals.broadcast()

    if (perturb > 0) {
      val mids = z_vals.get("...,1:").add(z_vals.get("...,:-1")).mul(.5)
      val upper = mids.concat(z_vals.get("...,-1:"), -1)
      val lower = z_vals.get("...,:1").concat(mids, -1)
      val t_rand = manager.randomUniform(0, 1, z_vals.getShape, DataType.FLOAT32)
      z_vals = lower.add(upper.sub(lower).mul(t_rand))
    }
    val pts = rays_o.expandDims(-2).add(rays_d.expandDims(-2).mul(z_vals.expandDims(-1)))

    val raw = if (train) trainer.forward(run_network(pts, viewdirs)) else trainer.evaluate(run_network(pts, viewdirs))
    val (rgb_map, disp_map, acc_map, weights, depth_map) = raw2outputs(raw, z_vals, rays_d)

    var rgb_map_1:NDArray = null
    var disp_map_1:NDArray = null
    var acc_map_1:NDArray = null

    if (args.N_importance > 0) {
      val z_vals_mid = z_vals.get("...,1:").add(z_vals.get("...,:-1")).mul(.5)
      val z_samples = sample_pdf(z_vals_mid, weights.get("...,1:-1"), args.N_importance, perturb == 0)
      //这里使用了自建函数，所以不需要手动停止梯度追踪
      val z_vals2 = z_vals.concat(z_samples, -1).sort(-1)
      val pts = rays_o.expandDims(-2).add(rays_d.expandDims(-2).mul(z_vals.expandDims(-1)))
      val raw=if(train)trainer_fine.forward(run_network(pts,viewdirs)) else trainer_fine.evaluate(run_network(pts,viewdirs))
      //val (rgb_map,disp_map,acc_map,weights,depth_map)
    }
  }

  private case class Embedder(include_input: Boolean, input_dims: Int, max_freq_log2: Int, num_freqs: Int, log_sampling: Boolean, periodic_fns: Array[NDArray => NDArray]) {
    val embed_fns = new ArrayBuffer[NDArray => NDArray]
    var out_dim = 0
    if (include_input) {
      embed_fns += (x => x)
      out_dim += input_dims
    }
    val subManager = manager.newSubManager()
    val freq_bands = if (log_sampling) subManager.linspace(0, max_freq_log2, num_freqs).toFloatArray.map(f => Math.pow(2, f).toFloat) else subManager.linspace(1, 1 << max_freq_log2, num_freqs).toFloatArray
    for (freq <- freq_bands) {
      for (p_fn <- periodic_fns) {
        embed_fns += (x => p_fn(x.mul(freq)))
        out_dim += input_dims
      }
    }
    subManager.close()


    def embeded(inputs: NDArray): NDArray = {
      if (embed_fns.isEmpty) {
        null
      } else {
        NDArrays.concat(new NDList(embed_fns.map(fn => fn(inputs)): _*), -1)
      }
    }
  }

  private def get_embedder(multires: Int, i: Int = 0): (NDArray => NDArray, Int) = {
    if (i == -1) {
      return ((x: NDArray) => x.get(), 3)
    }
    val embedder_obj = Embedder(true, 3, multires - 1, multires, true, Array(x => x.sin(), x => x.cos()))
    (embedder_obj.embeded(_), embedder_obj.out_dim)
  }

  private def init_nerf_block(D: Int = 8, W: Int = 256, output_ch: Int = 4, skips: Array[Int] = Array(4), use_viewdirs: Boolean = false): Block = {
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
        override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow(), x.get(1).singletonOrThrow())
      }, List[Block](input_viewdirs, alpha_out).asJava))
      //输出：NDList尺寸为2，其内容分别为RGB和density
    } else {
      block.add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
        override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow(), x.get(1).singletonOrThrow())
      }, List[Block](Linear.builder().setUnits(3).build(), Linear.builder().setUnits(1).build()).asJava))
      //输出：NDList尺寸为2，其内容分别为RGB和density
    }
  }

  def run_network(inputs: NDArray, viewdirs: NDArray): NDList = {
    //inputs第一维是batch，第二维是一条ray上的所有采样点，第三维才是点的内容
    //viewdirs第一维是batch，第二维是1，第三维是一条ray的方向

    val embeded = new NDList(embed_fn(inputs))
    if (viewdirs != null) {
      embeded.add(embeddirs_fn(viewdirs).tile(1, inputs.getShape.get(1)))
    }

    embeded
    //该输出可直接输入网络
  }

  def raw2outputs(raw: NDList, z_vals: NDArray, rays_d: NDArray): (NDArray, NDArray, NDArray, NDArray, NDArray) = {
    //raw内有两项，分别是颜色，密度
    //它们的0维度都是batch

    val manager = z_vals.getManager
    val dists = z_vals.get("...,1:").sub(z_vals.get("...,:-1")).concat(manager.create(1e10f).broadcast(new Shape(z_vals.getShape.get(0), 1)), -1).mul(rays_d.norm(Array(-1), true))
    val rgb = raw.get(0).getNDArrayInternal.sigmoid()

    var alpha: NDArray = raw.get(1)
    if (args.raw_noise_std > 0) {
      alpha = alpha.add(manager.randomNormal(alpha.getShape, DataType.FLOAT32))
    }
    alpha = alpha.getNDArrayInternal.relu().neg().mul(dists)

    var weights = alpha.cumSum(-1).get("...,:-1")
    alpha = NDArrays.sub(1, alpha.exp())
    weights = manager.ones(new Shape(weights.getShape.get(0), 1), DataType.FLOAT32).concat(weights.exp(), -1).mul(alpha)

    var rgb_map = weights.expandDims(-1).mul(rgb).sum(Array(-2))

    val depth_map = weights.mul(z_vals).sum(Array(-1))
    val acc_map = weights.sum(Array(-1))
    val disp_map = NDArrays.div(1, depth_map.div(acc_map).maximum(1e-10))

    if (args.white_bkgd) {
      rgb_map = rgb_map.add(NDArrays.sub(1, acc_map).expandDims(-1))
    }
    (rgb_map, disp_map, acc_map, weights, depth_map)
  }

  def searchShorted(shorted_sequence: Array[Float], values: Array[Float], batchNum: Int, N_samples: Int, shorted_samples: Int): Array[Array[Int]] = {
    (0 until batchNum).map { i =>
      var idx = 0
      (0 until N_samples).map { j =>
        while (values(i * N_samples + j) <= shorted_sequence(i * N_samples + idx) && idx < shorted_samples) {
          idx += 1
        }
        idx
      }.toArray
    }.toArray
  }

  def gather(params: Array[Float], indices: Array[Int], batchNum: Int, N_samples: Int): Array[Array[Float]] = {
    (0 until batchNum).map { i =>
      (0 until N_samples).map { j =>
        params(i * N_samples + indices(i * N_samples + j))
      }.toArray
    }.toArray
  }

  def sample_pdf(bins: NDArray, weights: NDArray, N_samples: Int, det: Boolean = false): NDArray = {
    val manager = weights.getManager
    val weights2 = weights.add(1e-5)
    val pdf = weights2.div(weights2.sum(Array(-1), true))
    val cdf = manager.zeros(new Shape(pdf.getShape.get(0), 1), DataType.FLOAT32).concat(pdf.cumSum(-1), -1)

    val u = if (det) manager.linspace(0, 1, N_samples).broadcast(Shape.update(cdf.getShape, cdf.getShape.dimension() - 1, N_samples))
    else manager.randomUniform(0, 1, Shape.update(cdf.getShape, cdf.getShape.dimension() - 1, N_samples))

    val inds = manager.create(searchShorted(cdf.toFloatArray, u.toFloatArray, cdf.getShape.get(0).toInt, N_samples, cdf.getShape.get(1).toInt))
    val below = inds.sub(1).maximum(0)
    val above = inds.minimum(cdf.getShape.tail() - 1)
    val inds_g = below.stack(above, -1)
    val cdf_g = manager.create(gather(cdf.toFloatArray, inds_g.toIntArray, cdf.getShape.get(0).toInt, N_samples))
    val bins_g = manager.create(gather(bins.toFloatArray, inds_g.toIntArray, bins.getShape.get(0).toInt, N_samples))

    var denom = cdf_g.get("...,1").sub(cdf_g.get("...,0"))
    denom = denom.getNDArrayInternal.where(denom.lt(1e-5), manager.ones(denom.getShape, DataType.FLOAT32))
    val t = (u.sub(cdf_g.get("...,0"))).div(denom)
    bins_g.get("...,0").add(t.mul(bins_g.get("...,1").sub(bins_g.get("...,0"))))
  }
}
