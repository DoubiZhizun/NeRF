package nerf_change

import ai.djl.modality.cv._
import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.ndarray.types._

import java.nio.file._
import java.util.function._
import scala.collection.mutable._

object load_llff {

  class int2ArrayPath extends IntFunction[Array[Path]] {
    override def apply(value: Int): Array[Path] = new Array[Path](value)
  }

  def _minify(basedir: String, factors: Array[Int] = new Array(0), resolutions: Array[(Int, Int)] = new Array(0)): Unit = {

    val imgdir_orig = Paths.get(basedir, "images")
    val imgs = for (f <- Files.list(imgdir_orig).toArray(new int2ArrayPath) if Array("JPG", "jpg", "png", "jpeg", "PNG").exists(ex => f.toString.endsWith(ex))) yield f

    def minifying(resizeArg: String, imgdir: Path, r: String): Unit = {
      print(s"Minifying $r " + basedir + "\n")
      Files.createDirectory(imgdir)
      for (f <- imgs) {
        val fd = Paths.get(imgdir.toString, f.toString.substring(imgdir_orig.toString.length))
        Files.copy(f, fd)
        val ext = f.toString.split("\\.").last
        val args = String.join(" ", "magick", "mogrify", "-resize", resizeArg, "-format", "png", fd.toString)
        print(args + "\n")
        val process = Runtime.getRuntime.exec(args)
        process.waitFor()
        if (ext != "png") {
          Files.delete(fd)
          print("Removed duplicates\n")
        }
      }
      print("Done\n")
    }

    for (r <- factors) {
      val name = s"images_${r}"
      val resizeArg = s"${100.toDouble / r}%"
      val imgdir = Paths.get(basedir, name)
      if (!Files.exists(imgdir)) {
        minifying(resizeArg, imgdir, r.toString)
      }
    }
    for (r <- resolutions) {
      val name = s"images_${r._2}x${r._1}"
      val resizeArg = s"${r._2}x${r._1}"
      val imgdir = Paths.get(basedir, name)
      if (!Files.exists(imgdir)) {
        minifying(resizeArg, imgdir, r.toString)
      }
    }
  }

  def load(basedir: String, manager: NDManager): NDArray = {
    val bytes = Files.readAllBytes(Paths.get(basedir, "poses_bounds.bin"))
    val array = new Array[Double](bytes.length / 8)
    for (i <- array.indices) {
      var longNum: Long = 0
      for (j <- 0 until 8) {
        longNum |= (bytes(i * 8 + j).toLong & 0xff) << j * 8
      }
      array(i) = java.lang.Double.longBitsToDouble(longNum)
    }
    manager.create(array, new Shape(array.length / 17, 17)).toType(DataType.FLOAT64, false)
  }

  def _load_data(basedir: String, factor: Int = 0, width: Int = 0, height: Int = 0, load_imgs: Boolean = true, manager: NDManager): (NDArray, NDArray, NDArray) = {
    //新增参数manager
    val poses_arr = load(basedir, manager)
    val poses = poses_arr.get(":,:-2").reshape(-1, 3, 5).transpose(1, 2, 0)
    val bds = poses_arr.get(":,-2:").transpose(1, 0)

    val img0 = (for (f <- Files.list(Paths.get(basedir, "images")).toArray(new int2ArrayPath) if f.toString.endsWith("JPG") || f.toString.endsWith("jpg") || f.toString.endsWith("png")) yield f).head
    val sh = ImageFactory.getInstance().fromFile(img0).toNDArray(manager).getShape

    var sfx = ""
    var factor2: Double = 0
    var width2: Int = 0
    var height2: Int = 0

    if (factor != 0) {
      sfx = s"_$factor"
      _minify(basedir, factors = Array(factor))
      factor2 = factor
    } else if (height != 0) {
      factor2 = sh.get(0) / height.toDouble
      width2 = (sh.get(1) / factor).toInt
      height2 = height
      _minify(basedir, resolutions = Array((height2, width2)))
      sfx = s"_${width2}x${height2}"
    } else if (width != 0) {
      factor2 = sh.get(1) / width.toDouble
      height2 = (sh.get(0) / factor).toInt
      width2 = width
      _minify(basedir, resolutions = Array((height2, width2)))
      sfx = s"_${width2}x${height2}"
    } else {
      factor2 = 1
    }

    val imgdir = Paths.get(basedir, "images" + sfx)
    if (!Files.exists(imgdir)) {
      print(imgdir.toString + " does not exist, returning\n")
      return (null, null, null)
    }

    val imgFiles = (for (f <- Files.list(imgdir).toArray(new int2ArrayPath) if f.toString.endsWith("JPG") || f.toString.endsWith("jpg") || f.toString.endsWith("png")) yield f).sortWith((x, y) => x.compareTo(y) < 0)
    if (poses.getShape.getShape.last != imgFiles.length) {
      print(s"Mismatch between imgs ${imgFiles.length} and poses ${poses.getShape.getShape.last} !!!!\n")
      return (null, null, null)
    }

    val sh2 = ImageFactory.getInstance().fromFile(imgFiles(0)).toNDArray(manager).getShape
    poses.set(new NDIndex(":2,4,:"), manager.create(Array(sh2.get(0), sh2.get(1)), new Shape(2, 1, 1)).toType(poses_arr.getDataType, false).broadcast(2, 1, poses.getShape.get(2)))
    poses.set(new NDIndex("2,4,:"), poses.get("2,4,:").div(factor2))

    if (!load_imgs) {
      return (poses, bds, null)
    }

    def imread(f: Path): NDArray = {
      ImageFactory.getInstance().fromFile(f).toNDArray(manager)
    }

    val imgs = NDArrays.stack(new NDList((for (f <- imgFiles) yield imread(f).get("...,:3").div(255)): _*), -1)

    print(s"Loaded image data ${imgs.getShape} ${poses.get(":,-1,0").toDoubleArray.mkString("(", ", ", ")")}\n")
    (poses, bds, imgs)
  }

  def normalize(x: NDArray): NDArray = {
    x.div(x.norm())
  }

  def cross(x: Array[Double], y: Array[Double]): Array[Double] = {
    val z = new Array[Double](3)
    z(0) = x(1) * y(2) - x(2) * y(1)
    z(1) = x(2) * y(0) - x(0) * y(2)
    z(2) = x(0) * y(1) - x(1) * y(0)
    z
  }

  def viewmatrix(z: NDArray, up: NDArray, pos: NDArray): NDArray = {
    val manager = z.getManager
    val vec2 = normalize(z)
    val vec1_arg = up
    val vec0 = normalize(manager.create(cross(vec1_arg.toDoubleArray, vec2.toDoubleArray)).toType(z.getDataType, false))
    val vec1 = normalize(manager.create(cross(vec2.toDoubleArray, vec0.toDoubleArray)).toType(z.getDataType, false))
    vec0.getNDArrayInternal.stack(new NDList(vec1, vec2, pos), 1)
  }

  def poses_avg(poses: NDArray): NDArray = {
    val hwf = poses.get("0,:3,-1:")
    val center = poses.get(":,:3,3").mean(Array(0))
    val vec2 = normalize(poses.get(":,:3,2").sum(Array(0)))
    val up = poses.get(":,:3,1").sum(Array(0))
    viewmatrix(vec2, up, center).concat(hwf, 1)
  }

  def render_path_spiral(c2w: NDArray, up: NDArray, rads: NDArray, focal: Double, zdelta: Double, zrate: Double, rots: Int, N: Int): Array[NDArray] = {
    val render_poses = new ArrayBuffer[NDArray]
    val manager = c2w.getManager
    val rads2 = rads.concat(manager.create(Array(1)).toType(c2w.getDataType, false))
    val hwf = c2w.get(":,4:5")
    for (theta <- manager.linspace(0, (2 * Math.PI * rots).toFloat, N + 1).get(":-1").toFloatArray) {
      val c = c2w.get(":3,:4").matMul(manager.create(Array(Math.cos(theta), -Math.sin(theta), -Math.sin(theta * zrate), 1)).toType(c2w.getDataType, false).mul(rads2))
      val z = normalize(c.sub(c2w.get(":3,:4").matMul(manager.create(Array(0, 0, -focal, 1)).toType(c2w.getDataType, false))))
      render_poses += viewmatrix(z, up, c).concat(hwf, 1)
    }
    render_poses.toArray
  }

  def inv(x: Array[Double]): Array[Array[Double]] = {
    val xSize = Math.sqrt(x.length).round.toInt
    val y = Array.fill(xSize)(new Array[Double](xSize))
    for (i <- 0 until xSize) {
      val temp = 1 / x(i * xSize + i)
      for (k <- i + 1 until xSize) {
        x(i * xSize + k) *= temp
      }
      for (k <- 0 until i) {
        y(i)(k) *= temp
      }
      y(i)(i) = temp
      for (j <- 0 until xSize) {
        if (i != j) {
          for (k <- i + 1 until xSize) {
            x(j * xSize + k) -= x(j * xSize + i) * x(i * xSize + k)
          }
          for (k <- 0 until i) {
            y(j)(k) -= x(j * xSize + i) * y(i)(k)
          }
          y(j)(i) = -x(j * xSize + i) * temp
        }
      }
    }
    y
  }

  def recenter_poses(poses: NDArray): NDArray = {
    val manager = poses.getManager
    val poses_ = poses.add(0)
    val bottom = manager.create(Array(0, 0, 0, 1), new Shape(1, 4)).toType(poses.getDataType, false)
    val c2w = poses_avg(poses).get(":3,:4").concat(bottom, -2)
    val bottom2 = bottom.reshape(new Shape(1, 1, 4)).broadcast(poses.getShape.get(0), 1, 4)
    val poses2 = poses.get(":,:3,:4").concat(bottom2, -2)
    val poses3 = manager.create(inv(c2w.toDoubleArray)).toType(poses.getDataType, false).matMul(poses2)
    poses_.set(new NDIndex(":,:3,:4"), poses3.get(":,:3,:4"))
    poses_
  }

  def spherify_poses(poses: NDArray, bds: NDArray): (NDArray, NDArray, NDArray) = {
    val manager = poses.getManager
    val p34_to_44 = (p: NDArray) => p.concat(manager.create(Array(0, 0, 0, 1), new Shape(1, 1, 4)).toType(poses.getDataType, false).broadcast(p.getShape.get(0), 1, 4), 1)

    val rays_d = poses.get(":,:3,2:3")
    val rays_o = poses.get(":,:3,3:4")

    def min_line_dist(rays_o: NDArray, rays_d: NDArray): NDArray = {
      val A_i = manager.eye(3).sub(rays_d.mul(rays_d.transpose(0, 2, 1)))
      val b_i = A_i.neg().matMul(rays_o)
      manager.create(inv(A_i.transpose(0, 2, 1).matMul(A_i).mean(Array(0)).toDoubleArray)).toType(rays_o.getDataType, false).neg().matMul(b_i.mean(Array(0))).squeeze()
    }

    val pt_minDist = min_line_dist(rays_o, rays_d)
    val center = pt_minDist
    val up = poses.get(":,:3,3").sub(center).mean(Array(0))

    val vec0 = normalize(up)
    val vec1 = normalize(manager.create(cross(Array(.1, .2, .3), vec0.toDoubleArray)).toType(rays_o.getDataType, false))
    val vec2 = normalize(manager.create(cross(vec0.toDoubleArray, vec1.toDoubleArray)).toType(rays_o.getDataType, false))
    val pos = center
    val c2w = vec1.getNDArrayInternal.stack(new NDList(vec2, vec0, pos), 1)

    val poses_reset = manager.create(inv(p34_to_44(c2w.reshape(1L +: c2w.getShape.getShape: _*)).toDoubleArray)).toType(poses.getDataType, false).matMul(p34_to_44(poses.get(":,:3,:4")))

    val rad = poses_reset.get(":,:3,3").square().sum(Array(-1)).mean().sqrt().getDouble()

    val sc = 1 / rad
    poses_reset.set(new NDIndex(":,:3,3"), poses_reset.get(":,:3,3").mul(sc))
    bds.muli(sc)
    val rad2 = rad * sc

    val centroid = poses_reset.get(":,:3,3").mean(Array(0))
    val zh = centroid.getDouble(2)
    val radCircle = Math.sqrt(rad2 * rad2 - zh * zh)
    val new_poses = new ArrayBuffer[NDArray]
    for (th <- manager.linspace(0, 2 * Math.PI.toFloat, 120).toFloatArray) {
      val camOrigin = manager.create(Array(radCircle * Math.cos(th), radCircle * Math.sin(th), zh)).toType(poses.getDataType, false)
      val up = manager.create(Array(0, 0, -1)).toType(poses.getDataType, false)

      val vec2 = normalize(camOrigin)
      val vec0 = normalize(manager.create(cross(vec2.toDoubleArray, up.toDoubleArray)).toType(poses.getDataType, false))
      val vec1 = normalize(manager.create(cross(vec2.toDoubleArray, vec0.toDoubleArray)).toType(poses.getDataType, false))
      val pos = camOrigin
      new_poses += vec0.getNDArrayInternal.stack(new NDList(vec1, vec2, pos), 1)
    }

    val new_poses2 = NDArrays.stack(new NDList(new_poses: _*)).concat(poses.get("0,:3,-1:").broadcast(new Shape(new_poses.length + 1, 3, 1)), -1)
    val poses_reset2 = poses_reset.get(":,:3,:4").concat(poses.get("0,:3,-1:").broadcast(new Shape(poses_reset.getShape.get(0), 3, 1)), -1)
    (poses_reset2, new_poses2, bds)
  }

  def percentile(x: Array[Double], xSize: Int, percent: Int): Array[Double] = {
    val y = new Array[Double](x.length / xSize)
    val xSlice = new Array[Double](xSize)
    val place = percent * (xSize - 1) / 100
    val rest = (percent * (xSize - 1) % 100) / 100.toDouble
    y.indices.foreach { i =>
      xSlice.indices.foreach { j =>
        xSlice(j) = x(j * y.length + i)
      }
      val xSlice2 = xSlice.sortWith(_ < _)
      y(i) = xSlice2(place)
      if (percent != 100) {
        y(i) += rest * (xSlice2(place + 1) - xSlice2(place))
      }
    }
    y
  }

  def load_llff_data(basedir: String, factor: Int = 8, recenter: Boolean = true, bd_factor: Double = .75, spherify: Boolean = false, path_zflat: Boolean = false, manager: NDManager): (NDArray, NDArray, NDArray, NDArray, Array[Int]) = {

    var (poses, bds, imgs) = _load_data(basedir, factor = factor, manager = manager)
    print(s"Loaded ${basedir} ${bds.min().getDouble()} ${bds.max().getDouble()}\n")

    poses = poses.get(":,1:2,:").concat(poses.get(":,0:1,:").neg(), 1).concat(poses.get(":,2:,:"), 1)
    poses = poses.transpose((poses.getShape.dimension() - 1) +: (0 until poses.getShape.dimension() - 1).toArray: _*)
    imgs = imgs.transpose((imgs.getShape.dimension() - 1) +: (0 until imgs.getShape.dimension() - 1).toArray: _*)
    val images = imgs
    bds = bds.transpose((bds.getShape.dimension() - 1) +: (0 until bds.getShape.dimension() - 1).toArray: _*)

    val sc = if (bd_factor == 0) 1 else 1 / (bds.min().getDouble() * bd_factor)
    poses.set(new NDIndex(":,:3,3"), poses.get(":,:3,3").mul(sc))
    bds.muli(sc)

    if (recenter) {
      poses = recenter_poses(poses)
    }

    var render_poses: NDArray = null

    if (spherify) {
      val (poses2, render_poses2, bds2) = spherify_poses(poses, bds)
      val p = poses.toDoubleArray

      poses = poses2
      render_poses = render_poses2
      bds = bds2
    } else {
      val c2w = poses_avg(poses)
      print(s"recentered ${c2w.getShape}\n")
      print(c2w.get(":3,:4"))

      val up = normalize(poses.get(":,:3,1").sum(Array(0)))

      val close_depth = bds.min().getDouble() * .9
      val inf_depth = bds.max().getDouble() * 5
      val dt = .75
      val mean_dz = 1 / ((1 - dt) / close_depth + dt / inf_depth)
      val focal = mean_dz

      val zdelta = close_depth * .2
      val tt = poses.get(":,:3,3").abs()
      val rads = manager.create(percentile(tt.toDoubleArray, tt.getShape.get(0).toInt, 90)).toType(poses.getDataType, false)
      //val rads = tt.percentile(90, Array(0))
      val c2w_path = c2w
      var N_views = 120
      var N_rots = 2
      if (path_zflat) {
        val zloc = -close_depth * .1
        c2w_path.set(new NDIndex(":3,3"), c2w_path.get(":3,3").add(c2w_path.get(":3,2").mul(zloc)))
        rads.set(new NDIndex(2), 0)
        N_rots = 1
        N_views /= 2
      }
      render_poses = NDArrays.stack(new NDList(render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate = .5, rots = N_rots, N = N_views): _*), 0)
    }

    val c2w = poses_avg(poses)
    print("Data:\n")
    print(s"${poses.getShape} ${images.getShape} ${bds.getShape}\n")

    val dists = c2w.get(":3,3").sub(poses.get(":,:3,3")).square().sum(Array(-1))
    val i_test = Array(dists.argMin().getLong().toInt)
    print(s"HOLDOUT view is ${i_test.head}\n")

    (images.toType(DataType.FLOAT32, false), poses.toType(DataType.FLOAT32, false), bds.toType(DataType.FLOAT32, false), render_poses.toType(DataType.FLOAT32, false), i_test)
  }
}