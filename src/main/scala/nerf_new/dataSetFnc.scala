package nerf_new

import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.training.dataset._
import nerf_new.load_llff._

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
    val o1 = rays_o2.get("...,1").div(rays_o2.get("...,2")).mul(-1 / (H / (2 * focal)))
    val o2 = NDArrays.div(2 * near, rays_o2.get("...,2")).add(1)

    val d0 = rays_d.get("...,0").div(rays_d.get("...,2")).sub(rays_o2.get("...,0").div(rays_o2.get("...,2"))).mul(-1 / (W / (2 * focal)))
    val d1 = rays_d.get("...,1").div(rays_d.get("...,2")).sub(rays_o2.get("...,1").div(rays_o2.get("...,2"))).mul(-1 / (H / (2 * focal)))
    val d2 = NDArrays.div(-2 * near, rays_o2.get("...,2"))

    (d0.getNDArrayInternal.stack(new NDList(d1, d2), -1), o0.getNDArrayInternal.stack(new NDList(o1, o2), -1))
  }

  def getDataSet(config: nerfConfig, manager: NDManager): (Dataset, Int, NDList) = {
    val subManager = manager.newSubManager()
    var (images, poses, bds, render_poses, i_test) = load_llff_data(config.datadir, 8, true, 1, false, manager = subManager)
    val hwf = poses.get("0,:3,-1").toFloatArray
    poses = poses.get(":,:3,:4")
    hwf(0) = hwf(0).toInt
    hwf(1) = hwf(1).toInt
    //????????????????????????????????????????????????0??????10????????????????????????????????????
    //i_test = Array(0, 10)
    //val i_train = (for (i <- 0 until images.getShape.get(0).toInt if !i_test.contains(i)) yield i).toArray
    var (rays_d_temp, rays_o) = get_rays_np(hwf(0).toInt, hwf(1).toInt, hwf(2), poses.get("...,:4"))
    rays_d_temp = rays_d_temp.reshape(-1, 1, 3)
    rays_o = rays_o.reshape(-1, 1, 3)


    val rays_d_norm = rays_d_temp.norm(Array(-1), true)
    val far = rays_d_norm.squeeze(-1)
    val near = subManager.zeros(far.getShape)
    val rays_d = rays_d_temp.div(rays_d_norm)
    val viewdirs = rays_d.expandDims(-2)
    val label = images.reshape(-1, 3)

    label.attach(manager)
    rays_d.attach(manager)
    rays_o.attach(manager)
    near.attach(manager)
    far.attach(manager)
    viewdirs.attach(manager)

    val dataSet = new nerfDataSet(rays_o, rays_d, near, far, viewdirs, label, config.N_rand)

    var (render_d_temp, render_o) = get_rays_np(hwf(0).toInt, hwf(1).toInt, hwf(2), poses.get("...,:4").concat(render_poses.get("...,:4"), 0))
    render_d_temp = render_d_temp.expandDims(-2)
    render_o = render_o.expandDims(-2)
    val render_d_norm = render_d_temp.norm(Array(-1), true)
    val render_far = render_d_norm.squeeze(-1)
    val render_near = subManager.zeros(render_far.getShape)
    val render_d = render_d_temp.div(render_d_norm)
    val render_viewdirs = render_d.expandDims(-2)

    render_o.attach(manager)
    render_d.attach(manager)
    render_near.attach(manager)
    render_far.attach(manager)
    render_viewdirs.attach(manager)

    subManager.close()
    //???????????????????????????
    (dataSet, label.getShape.get(0).toInt, new NDList(render_o, render_d, render_near, render_far, render_viewdirs))
  }
}