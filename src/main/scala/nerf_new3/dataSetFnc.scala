package nerf_new3

import ai.djl.modality.cv.Image
import ai.djl.modality.cv.util.NDImageUtils
import ai.djl.ndarray._
import ai.djl.training.dataset._
import nerf_new2.load_llff._

object dataSetFnc {

  def getDataSet(config: nerfConfig, manager: NDManager): (Dataset, Array[Float], NDList) = {
    val subManager = manager.newSubManager()
    var (images, poses, bds, render_poses, i_test) = load_llff_data(config.datadir, 6, true, .75, false, manager = subManager)
    val hwf = poses.get("0,:3,-1").toFloatArray
    poses = poses.get(":,:3,:4")
    hwf(2) = hwf(2) * 384 / hwf(0)
    hwf(0) = 384
    hwf(1) = 512

    val label = NDImageUtils.resize(images, 512, 384, Image.Interpolation.BILINEAR)
    val c2w = poses.get("...,:3")
    val rays_o = poses.get("...,3")
    val render_c2w = c2w.concat(render_poses.get("...,:3"), 0)
    val render_rays_o = rays_o.concat(render_poses.get("...,3"), 0)
    label.attach(manager)
    c2w.attach(manager)
    rays_o.attach(manager)
    render_c2w.attach(manager)
    render_rays_o.attach(manager)
    subManager.close()
    val dataSet = new nerfDataSet(c2w, rays_o, label)
    (dataSet, hwf, new NDList(render_c2w, render_rays_o))
  }
}