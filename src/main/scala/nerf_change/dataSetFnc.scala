package nerf_change

import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.training.dataset._
import load_llff._

object dataSetFnc {

  def get_rays_np(H: Int, W: Int, focal: Double, c2w: NDArray): (NDArray, NDArray) = {
    val manager = c2w.getManager
    val i = manager.arange(0, W, 1, DataType.FLOAT32).reshape(1, W).repeat(0, H)
    val j = manager.arange(0, H, 1, DataType.FLOAT32).reshape(H, 1).repeat(1, W)
    val dirs = NDArrays.stack(new NDList(i.sub(W * .5).div(focal), j.sub(H * .5).div(focal).neg(), manager.ones(i.getShape).neg()), -1)
    val rays_d = dirs.expandDims(-2).expandDims(-2).mul(c2w.get(":,:,:3")).sum(Array(-1))
    ndc_rays(H, W, focal, 1, rays_d.transpose(2, 0, 1, 3), c2w.get(":,:,-1").broadcast(rays_d.getShape).transpose(2, 0, 1, 3))
  }

  def ndc_rays(H: Int, W: Int, focal: Double, near: Double, rays_d: NDArray, rays_o: NDArray): (NDArray, NDArray) = {
    val t = rays_o.get("...,2:3").add(near).neg().div(rays_d.get("...,2:3"))
    val rays_o2 = rays_o.add(t.mul(rays_d))

    val o0 = rays_o2.get("...,0").div(rays_o2.get("...,2")).mul(-1 / (W / (2 * focal)))
    val o1 = rays_o2.get("...,1").div(rays_o2.get("...,2")).mul(-1 / (W / (2 * focal)))
    val o2 = NDArrays.div(1 + 2 * near, rays_o2.get("...,2"))

    val d0 = rays_d.get("...,0").div(rays_d.get("...,2")).sub(rays_o2.get("...,0").div(rays_o2.get("...,2"))).mul(-1 / (W / (2 * focal)))
    val d1 = rays_d.get("...,1").div(rays_d.get("...,2")).sub(rays_o2.get("...,1").div(rays_o2.get("...,2"))).mul(-1 / (W / (2 * focal)))
    val d2 = NDArrays.div(-2 * near, rays_o2.get("...,2"))

    (d0.getNDArrayInternal.stack(new NDList(d1, d2), -1), o0.getNDArrayInternal.stack(new NDList(o1, o2), -1))
  }

  def getDataSet(config: nerfConfig, manager: NDManager): (Dataset, Dataset, Int, NDList) = {
    val subManager = manager.newSubManager()
    var (images, poses, bds, render_poses, i_test) = load_llff_data(config.datadir, 8, true, .75, false, manager = subManager)
    val hwf = poses.get("0,:3,-1").toFloatArray
    poses = poses.get(":,:3,:4")
    hwf(0) = hwf(0).toInt
    hwf(1) = hwf(1).toInt
    //该数据集中共有二十个数据，抓数据0号和10号做测试集，其他做训练集
    //i_test = Array(0, 10)
    //val i_train = (for (i <- 0 until images.getShape.get(0).toInt if !i_test.contains(i)) yield i).toArray
    var trainImages = images.get("1:10").concat(images.get("11:"), 0).reshape(-1, 3)
    var testImages = images.get(":1").concat(images.get("10:11"), 0).reshape(-1, 3)
    trainImages = trainImages.concat(trainImages, 1)
    testImages = testImages.concat(testImages, 1)
    val trainPoses = poses.get("1:10").concat(poses.get("11:"), 0)
    val testPoses = poses.get(":1").concat(poses.get("10:11"), 0)
    var (rays_d_train_temp, rays_o_train) = get_rays_np(hwf(0).toInt, hwf(1).toInt, hwf(2), trainPoses)
    var (rays_d_test_temp, rays_o_test) = get_rays_np(hwf(0).toInt, hwf(1).toInt, hwf(2), testPoses)
    rays_d_train_temp = rays_d_train_temp.reshape(-1, 3)
    rays_o_train = rays_o_train.reshape(-1, 3)
    rays_d_test_temp = rays_d_test_temp.reshape(-1, 3)
    rays_o_test = rays_o_test.reshape(-1, 3)


    val rays_d_train_norm = rays_d_train_temp.norm(Array(-1), true)
    val bounds_train = subManager.zeros(rays_d_train_norm.getShape).concat(rays_d_train_norm, 1)
    val rays_d_train = rays_d_train_temp.div(rays_d_train_norm)
    val rays_d_test_norm = rays_d_test_temp.norm(Array(-1), true)
    val bounds_test = subManager.zeros(rays_d_test_norm.getShape).concat(rays_d_test_norm, 1)
    val rays_d_test = rays_d_test_temp.div(rays_d_test_norm)

    trainImages.attach(manager)
    testImages.attach(manager)
    rays_d_train.attach(manager)
    rays_d_test.attach(manager)
    rays_o_train.attach(manager)
    rays_o_test.attach(manager)
    bounds_train.attach(manager)
    bounds_test.attach(manager)

    val trainSet = new ArrayDataset.Builder().setData(rays_o_train, rays_d_train, bounds_train, rays_d_train).optLabels(trainImages).setSampling(config.N_rand, true).build()
    //new nerfDataSet(rays_o_train, rays_d_train, bounds_train, rays_d_train, trainImages, config.N_rand)
    val testSet = new ArrayDataset.Builder().setData(rays_o_test, rays_d_test, bounds_test, rays_d_test).optLabels(testImages).setSampling(config.N_rand, true).build()
    //new nerfDataSet(rays_o_test, rays_d_test, bounds_test, rays_d_test, testImages, config.N_rand)

    val (render_d_temp, render_o) = get_rays_np(hwf(0).toInt, hwf(1).toInt, hwf(2), poses.get("...,:4").concat(render_poses.get("...,:4"), 0))
    val render_d_norm = render_d_temp.norm(Array(-1), true)
    val render_bounds = subManager.zeros(render_d_norm.getShape).concat(render_d_norm, -1)
    val render_d = render_d_temp.div(render_d_norm)

    render_o.attach(manager)
    render_d.attach(manager)
    render_bounds.attach(manager)

    subManager.close()
    //其他数据暂时先不用
    (trainSet, testSet, trainImages.getShape.get(0).toInt, new NDList(render_o, render_d, render_bounds))
  }
}