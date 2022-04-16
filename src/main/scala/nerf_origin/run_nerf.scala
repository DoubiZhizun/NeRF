package nerf_origin

import ai.djl._
import scopt._
import ai.djl.ndarray._
import ai.djl.engine._
import ai.djl.ndarray.index._
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.training._
import ai.djl.training.dataset._
import ai.djl.training.loss._

import scala.util.Random._
import load_llff._
import run_nerf_helpers._

import java.nio.file._

object run_nerf {

  def render_rays(ray_batch: Batch, lindisp: Boolean, N_samples: Int, perturb: Double): Unit = {
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
    val viewdirs = ray_batch.getData.get(4)

    val manager = ray_batch.getManager
    val t_vals = manager.linspace(0, 1, N_samples)
    var z_vals: NDArray = null
    if (!lindisp) {
      //在near和far之间均匀采样
      z_vals = near.mul(NDArrays.sub(1, t_vals)).add(far.mul(t_vals))
    } else {
      z_vals = NDArrays.div(1, NDArrays.div(1, near).mul(NDArrays.sub(1, t_vals)).add(NDArrays.div(1, far).mul(t_vals)))
    }

    if (perturb > 0) {
      val mids = z_vals.get("...,1:").add(z_vals.get("...,:-1")).mul(.5)
      val upper = mids.concat(z_vals.get("...,-1:"), -1)
      val lower = z_vals.get("...,:1").concat(mids, -1)
      val t_rand = manager.randomUniform(0, 1, z_vals.getShape)
      z_vals = lower.add(upper.sub(lower).mul(t_rand))
    }
    val pts = rays_o.reshape(N_rays, 1, 3).add(rays_d.reshape(N_rays, 1, 3).mul(z_vals.reshape(N_rays, N_samples, 1)))

  }

  def batchify(fn: NDArray => NDArray, chunk: Int): NDArray => NDArray = {
    if (chunk <= 0) {
      fn
    } else {
      x => NDArrays.concat(new NDList((for (i <- 0 until(x.getShape.get(0).toInt, chunk)) yield fn(x.get(s"${i}:${i + chunk}"))): _*), 0)
    }
  }

  def run_network(inputs: NDArray, viewdirs: NDArray, fn: NDArray => NDArray, embed_fn: NDArray => NDArray, embeddirs_fn: NDArray => NDArray, netchunk: Int = 1024 * 64): NDList = {
    val inputs_flat = inputs.reshape(-1, inputs.getShape.get(inputs.getShape.dimension() - 1))

    val embeded = new NDList(embed_fn(inputs_flat))
    if (viewdirs != null) {
      val input_dirs = viewdirs.reshape(viewdirs.getShape.getShape :+ 1l: _*).broadcast(inputs.getShape)
      val input_dirs_flat = input_dirs.reshape(-1, input_dirs.getShape.get(input_dirs.getShape.dimension() - 1))
      embeded.add(embeddirs_fn(input_dirs_flat))
    }

    val output_fn = batchify(fn, netchunk)
    val output_flat = new NDList()
    for (i <- 0 until embeded.size()) {
      output_flat.add(output_fn(embeded.get(i)))
    }

    output_flat
  }

  def create_nerf(args: Config, manager: NDManager): Block = {
    val (embed_fn, input_ch) = get_embedder(args.multires, args.i_embed)
    var input_ch_views = 0
    var embeddirs_fn: NDArray => NDArray = null
    if (args.use_viewdirs) {
      val temp = get_embedder(args.multires_views, args.i_embed)
      input_ch_views = temp._2
      embeddirs_fn = temp._1
    }
    val block = init_nerf_model(D = args.netdepth, W = args.netwidth, output_ch = 4, skips = Array(4), use_viewdirs = args.use_viewdirs)
    var block_fine: Block = null
    if (args.N_importance > 0) {
      block_fine = init_nerf_model(D = args.netdepth_fine, W = args.netwidth_fine, output_ch = 4, skips = Array(4), use_viewdirs = args.use_viewdirs)
      block_fine.initialize(manager, DataType.FLOAT32, new Shape(input_ch), new Shape(input_ch_views))
    }

    def network_query_fn(inputs: NDArray, viewdirs: NDArray, network_fn: NDArray => NDArray): NDList = run_network(inputs, viewdirs, network_fn, embed_fn, embeddirs_fn, args.netchunk)

    case class render_kwargs(network_query_fn: (NDArray, NDArray, NDArray => NDArray) => NDList,
                             perturb: Double,
                             N_importance: Int,
                             network_fine: (NDList, Boolean) => NDList,
                             N_samples: Int,
                             network_fn: (NDList, Boolean) => NDList,
                             use_viewdirs: Boolean,
                             write_bkgd: Boolean,
                             raw_noise_std: Double)
    val ps = new ParameterStore(manager, false)
    val render_kwargs_train = render_kwargs(network_query_fn, args.perturb, args.N_importance, block_fine.forward(ps, _, _), args.N_samples, block.forward(ps, _, _), args.use_viewdirs, args.white_bkgd, args.raw_noise_std)


    block
  }


  def config_parser(): OptionParser[Config] = {
    val parser = new OptionParser[Config]("scopt") {
      head("scopt", "3.x")

      opt[String]("expname").action((x, c) => c.copy(expname = x)).text("experiment name")
      opt[String]("basedir").action((x, c) => c.copy(basedir = x)).text("where to store ckpts and logs")
      opt[String]("datadir").action((x, c) => c.copy(datadir = x)).text("input data directory")
      opt[Int]("netdepth").action((x, c) => c.copy(netdepth = x)).text("layers in network")
      opt[Int]("netwidth").action((x, c) => c.copy(netwidth = x)).text("channels per layer")
      opt[Int]("netdepth_fine").action((x, c) => c.copy(netdepth_fine = x)).text("layers in fine network")
      opt[Int]("netwidth_fine").action((x, c) => c.copy(netwidth_fine = x)).text("channels per layer in fine network")
      opt[Int]("N_rand").action((x, c) => c.copy(N_rand = x)).text("batch size (number of random rays per gradient step)")
      opt[Double]("lrate").action((x, c) => c.copy(lrate = x)).text("learning rate")
      opt[Int]("lrate_decay").action((x, c) => c.copy(lrate_decay = x)).text("exponential learning rate decay (in 1000s)")
      opt[Int]("chunk").action((x, c) => c.copy(chunk = x)).text("number of rays processed in parallel, decrease if running out of memory")
      opt[Int]("netchunk").action((x, c) => c.copy(netchunk = x)).text("number of pts sent through network in parallel, decrease if running out of memory")
      opt[Boolean]("no_batching").action((x, c) => c.copy(no_batching = x)).text("only take random rays from 1 image at a time")
      opt[Boolean]("no_reload").action((x, c) => c.copy(no_reload = x)).text("do not reload weights from saved ckpt")
      opt[String]("ft_path").action((x, c) => c.copy(ft_path = x)).text("specific weights npy file to reload for coarse network")
      opt[Int]("random_seed").action((x, c) => c.copy(random_seed = x)).text("fix random seed for repeatability")
      opt[Int]("precrop_iters").action((x, c) => c.copy(precrop_iters = x)).text("number of steps to train on central crops")
      opt[Double]("precrop_frac").action((x, c) => c.copy(precrop_frac = x)).text("fraction of img taken for central crops")
      opt[Int]("N_samples").action((x, c) => c.copy(N_samples = x)).text("number of coarse samples per ray")
      opt[Int]("N_importance").action((x, c) => c.copy(N_importance = x)).text("number of additional fine samples per ray")
      opt[Double]("perturb").action((x, c) => c.copy(perturb = x)).text("set to 0. for no jitter, 1. for jitter")
      opt[Boolean]("use_viewdirs").action((x, c) => c.copy(use_viewdirs = x)).text("use full 5D input instead of 3D")
      opt[Int]("i_embed").action((x, c) => c.copy(i_embed = x)).text("set 0 for default positional encoding, -1 for none")
      opt[Int]("multires").action((x, c) => c.copy(multires = x)).text("log2 of max freq for positional encoding (3D location)")
      opt[Int]("multires_views").action((x, c) => c.copy(multires_views = x)).text("log2 of max freq for positional encoding (2D direction)")
      opt[Double]("raw_noise_std").action((x, c) => c.copy(raw_noise_std = x)).text("std dev of noise added to regularize sigma_a output, 1e0 recommended")
      opt[Boolean]("render_only").action((x, c) => c.copy(render_only = x)).text("do not optimize, reload weights and render out render_poses path")
      opt[Boolean]("render_test").action((x, c) => c.copy(render_test = x)).text("render the test set instead of render_poses path")
      opt[Int]("render_factor").action((x, c) => c.copy(render_factor = x)).text("downsampling factor to speed up rendering, set 4 or 8 for fast preview")
      opt[String]("dataset_type").action((x, c) => c.copy(dataset_type = x)).text("options: llff / blender / deepvoxels")
      opt[Int]("testskip").action((x, c) => c.copy(testskip = x)).text("will load 1/N images from test/val sets, useful for large datasets like deepvoxels")
      opt[String]("shape").action((x, c) => c.copy(shape = x)).text("options : armchair / cube / greek / vase")
      opt[Boolean]("white_bkgd").action((x, c) => c.copy(white_bkgd = x)).text("set to render synthetic data on a white bkgd (always use for dvoxels)")
      opt[Boolean]("half_res").action((x, c) => c.copy(half_res = x)).text("load blender synthetic data at 400x400 instead of 800x800")
      opt[Int]("factor").action((x, c) => c.copy(factor = x)).text("downsample factor for LLFF images")
      opt[Boolean]("no_ndc").action((x, c) => c.copy(no_ndc = x)).text("do not use normalized device coordinates (set for non-forward facing scenes)")
      opt[Boolean]("lindisp").action((x, c) => c.copy(lindisp = x)).text("sampling linearly in disparity rather than depth")
      opt[Boolean]("spherify").action((x, c) => c.copy(spherify = x)).text("set for spherical 360 scenes")
      opt[Int]("llffhold").action((x, c) => c.copy(llffhold = x)).text("will take every 1/N images as LLFF test set, paper uses 8")
      opt[Int]("i_print").action((x, c) => c.copy(i_print = x)).text("frequency of console printout and metric loggin")
      opt[Int]("i_img").action((x, c) => c.copy(i_img = x)).text("frequency of tensorboard image logging")
      opt[Int]("i_weights").action((x, c) => c.copy(i_weights = x)).text("frequency of weight ckpt saving")
      opt[Int]("i_testset").action((x, c) => c.copy(i_testset = x)).text("frequency of testset saving")
      opt[Int]("i_video").action((x, c) => c.copy(i_video = x)).text("frequency of render_poses video saving")

      val deviceRead = Read.reads {
        _.toLowerCase() match {
          case "cpu" => Device.cpu()
          case "gpu" => Device.gpu()
          case s =>
            throw new IllegalArgumentException("'" + s + "' is not a device.")
        }
      }

      opt[Device]("device")(deviceRead).action((x, c) => c.copy(device = x)).text("options: cpu / gpu")

      help('h', "help").text("show this help message and exit")
    }
    parser
  }

  def train(cmdArgs: Array[String]): Unit = {
    val parser = config_parser()
    val args = parser.parse(cmdArgs, Config()).head
    val manager = NDManager.newBaseManager(args.device)

    if (args.random_seed >= 0) {
      print(s"Fixing random seed ${args.random_seed}\n")
      Engine.getInstance().setRandomSeed(args.random_seed)
      setSeed(args.random_seed)
    }

    var images, poses, bds, render_poses: NDArray = null
    var hwf: Array[Float] = null
    var i_test, i_val, i_train: Array[Int] = null
    var near, far: Double = 0

    if (args.dataset_type == "llff") {
      val load = load_llff_data(args.datadir, args.factor, recenter = true, bd_factor = .75, spherify = args.spherify, manager = manager)
      images = load._1
      bds = load._3
      render_poses = load._4
      i_test = load._5
      val hwfTemp = load._2.get("0,:3,-1")
      hwf = hwfTemp.toFloatArray
      poses = load._2.get(":,:3,:4")
      print(s"load llff ${images.getShape} ${render_poses.getShape} ${hwf.mkString("(", ",", ")")} ${args.datadir}\n")

      if (args.llffhold > 0) {
        print(s"Auto LLFF holdout ${args.llffhold}\n")
        i_test = (0 until(images.getShape.get(0).toInt, args.llffhold)).toArray
      }

      i_val = i_test
      i_train = (for (i <- 0 until images.getShape.get(0).toInt if !i_val.contains(i)) yield i).toArray

      print("DEFINING BOUNDS\n")
      if (args.no_ndc) {
        near = bds.min().getDouble() * .9
        far = bds.max().getDouble()
      } else {
        near = 0
        far = 1
      }
      print(s"NEAR FAR $near $far\n")
      load._2.close()
      hwfTemp.close()
    }

    hwf(0) = hwf(0).toInt
    hwf(1) = hwf(1).toInt

    if (args.render_test) {
      render_poses.close()
      val shape = poses.getShape.getShape
      render_poses = manager.create(new Shape(i_test.length.toLong +: shape.slice(1, shape.length): _*), DataType.FLOAT32)
      for (i <- i_test.indices) {
        val temp = poses.get(s"${i_test(i)}:${i_test(i) + 1}")
        render_poses.set(new NDIndex(i), temp)
        temp.close()
      }
    }

    val basedir = args.basedir
    val expname = args.expname

    val expPath = Paths.get(basedir, expname)
    if (!Files.exists(expPath)) {
      Files.createDirectories(expPath)
    }
    //No arg files

    create_nerf(args, manager)
  }
}