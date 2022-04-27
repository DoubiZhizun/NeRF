package nerf_new3

import ai.djl.engine._
import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.training.loss._


class nerf(config: nerfConfig) {

  val loss = Loss.l2Loss("L2Loss", 1)

  def predict(H: Int, W: Int, focal: Float, c2w: NDArray, rays_o: NDArray, near: Float = 0, far: Float = 1): NDArray = {
    forward(H, W, focal, c2w, rays_o, near, far, false)
  }

  def train(H: Int, W: Int, focal: Float, c2w: NDArray, rays_o: NDArray, near: Float = 0, far: Float = 1, images: NDArray): Float = {
    //label：尺寸(H, W, 3)
    val collector = Engine.getInstance().newGradientCollector()
    val rgb_out = forward(H, W, focal, c2w, rays_o, near, far, true)
    val lossValue = loss.evaluate(new NDList(rgb_out), new NDList(images))
    collector.backward(lossValue)
    collector.close()
    config.ps.updateAllParameters()
    lossValue.getFloat()
  }

  def positionCode(input: NDArray, L: Int): NDArray = {
    //sin cos位置编码
    //input的最高维度是归一化过的
    val output = new NDList(L * 2)
    var factor = Math.PI
    for (_ <- 0 until L) {
      val inputMulFactor = input.mul(factor)
      output.add(inputMulFactor.sin())
      output.add(inputMulFactor.cos())
      factor *= 2
    }
    input.getNDArrayInternal.concat(output, -1)
  }


  def get_rays_np(H: Int, W: Int, focal: Float, IvH: Int, IvW: Int, c2w: NDArray): (NDArray, NDArray) = {
    //生成光线的方向的算法
    //输入：
    //H：图像高（像素点数）
    //W：图像宽（像素点数）
    //focal：相机焦距（物理焦距/像素在感光芯片上的物理距离）
    //IvH：特征图高方向采样点数
    //IvW：特征图宽方向采样点数
    //c2w：描述相机位置的正交矩阵，尺寸(3, 3)，三行（x、y、z）的方向分别是：
    //z：相机针对方向的反方向（即相机对准-z）
    //x：使用相机拍照时右手的方向
    //y：使用相机拍照时正上的方向
    val manager = c2w.getManager
    val h = (H - 1f) / 2 / focal
    val w = (W - 1f) / 2 / focal
    val i = manager.linspace(-w, w, IvW).reshape(1, IvW).repeat(0, IvH)
    val j = manager.linspace(-h, h, IvH).reshape(IvH, 1).repeat(1, IvW)
    val dirs = i.getNDArrayInternal.stack(new NDList(j.neg(), manager.full(i.getShape, -1, c2w.getDataType)), -1)
    val i_full = manager.linspace(-w, w, W).reshape(1, W).repeat(0, H)
    val j_full = manager.linspace(-h, h, H).reshape(H, 1).repeat(1, W)
    val dirs_full = i_full.getNDArrayInternal.stack(new NDList(j_full.neg(), manager.full(i_full.getShape, -1, c2w.getDataType)), -1)
    val c2wTrans = c2w.transpose(Array(1, 0): _*)
    (dirs.matMul(c2wTrans), dirs_full.matMul(c2wTrans))
    //返回：
    //rays_d：光线的方向，尺寸(IvH, IvW, 3)
    //rays_d_full：光线的方向完整版，尺寸(H, W, 3)
  }

  def ndc_rays(H: Int, W: Int, focal: Float, rays_o: NDArray, rays_d: NDArray, rays_d_full: NDArray): (NDArray, NDArray, NDArray) = {
    //对光线做ndc变换
    //近端平面恒定认为是-1
    //H、W、focal和rays_d同上
    //rays_o：光线的原点，尺寸(1, 1, 3)
    val manager = rays_o.getManager
    val rays_d2 = rays_d.get(new NDIndex().addAllDim(2).addIndices(2))
    val rays_d_full2 = rays_d_full.get(new NDIndex().addAllDim(2).addIndices(2))
    val t = rays_o.get(new NDIndex().addAllDim(2).addIndices(2)).add(1).div(rays_d2)
    val t_full = rays_o.get(new NDIndex().addAllDim(2).addIndices(2)).add(1).div(rays_d_full2)
    val rays_d0 = rays_d.get(new NDIndex().addAllDim(2).addIndices(0))
    val rays_d1 = rays_d.get(new NDIndex().addAllDim(2).addIndices(1))
    val rays_d_full0 = rays_d_full.get(new NDIndex().addAllDim(2).addIndices(0))
    val rays_d_full1 = rays_d_full.get(new NDIndex().addAllDim(2).addIndices(1))
    val rays_o0 = rays_o.get(new NDIndex().addAllDim(2).addIndices(0)).sub(t.mul(rays_d0))
    val rays_o1 = rays_o.get(new NDIndex().addAllDim(2).addIndices(1)).sub(t.mul(rays_d1))
    val rays_o_full0 = rays_o.get(new NDIndex().addAllDim(2).addIndices(0)).sub(t_full.mul(rays_d_full0))
    val rays_o_full1 = rays_o.get(new NDIndex().addAllDim(2).addIndices(1)).sub(t_full.mul(rays_d_full1))
    //将所有光线的原点映射到z == -1（即近端平面）上

    val o0 = rays_o0.mul(1 / (W / (2 * focal)))
    val o1 = rays_o1.mul(1 / (H / (2 * focal)))
    val o2 = manager.full(o0.getShape, -1, rays_o.getDataType)

    val d0 = rays_d0.div(rays_d2).add(rays_o0).mul(-1 / (W / (2 * focal)))
    val d1 = rays_d1.div(rays_d2).add(rays_o1).mul(-1 / (H / (2 * focal)))
    val d2 = manager.full(rays_o0.getShape, 2, rays_o.getDataType)

    val d_full0 = rays_d_full0.div(rays_d_full2).add(rays_o_full0).mul(-1 / (W / (2 * focal)))
    val d_full1 = rays_d_full1.div(rays_d_full2).add(rays_o_full1).mul(-1 / (H / (2 * focal)))
    val d_full2 = manager.full(rays_o_full0.getShape, 2, rays_o.getDataType)

    (o0.getNDArrayInternal.stack(new NDList(o1, o2), -1), d0.getNDArrayInternal.stack(new NDList(d1, d2), -1), d_full0.getNDArrayInternal.stack(new NDList(d_full1, d_full2), -1))
    //返回：
    //rays_o：尺寸(IvH, IvW, 3)
    //rays_d：尺寸(IvH, IvW, 3)
    //rays_d_full：尺寸(H, W, 3)
  }

  val from1 = new NDIndex("...,1:,:")
  val to1 = new NDIndex("...,:1,:")
  val fromNeg1 = new NDIndex("...,-1:,:")
  val toNeg1 = new NDIndex("...,:-1,:")

  val getSample = if (config.lindisp) (t_vals: NDArray, near: NDArray, far: NDArray) => NDArrays.div(1, NDArrays.div(1, near).mul(NDArrays.sub(1, t_vals)).add(NDArrays.div(1, far).mul(t_vals)))
  else (t_vals: NDArray, near: NDArray, far: NDArray) => near.mul(t_vals.sub(1).neg()).add(far.mul(t_vals))

  var givePerterb = if (config.perterb) (z_vals: NDArray) => {
    val manager = z_vals.getManager
    val mids = z_vals.get(from1).add(z_vals.get(toNeg1)).mul(.5)
    val upper = mids.concat(z_vals.get(fromNeg1), -2)
    val lower = z_vals.get(to1).concat(mids, -2)
    val t_rand = manager.randomUniform(0, 1, z_vals.getShape)
    lower.add(upper.sub(lower).mul(t_rand))
  } else (z_vals: NDArray) => z_vals

  def getInput(rays_o: NDArray, rays_d: NDArray, near: NDArray, far: NDArray, near_full: NDArray, far_full: NDArray): (NDArray, NDArray) = {
    //为网络准备输入
    //rays_o：尺寸(IvH, IvW, 3)或(1, 1, 3)
    //rays_d：尺寸(IvH, IvW, 3)
    //near：尺寸(IvH, IvW, 1)
    //far：尺寸(IvH, IvW, 1)
    //near_full：尺寸(H, W, 1)
    //far_full：尺寸(H, W, 1)
    val manager = rays_o.getManager
    val t_vals = manager.linspace(0, 1, config.N_samples)
    val z_vals = givePerterb(getSample(t_vals, near, far).expandDims(-1))
    (rays_o.expandDims(-2).add(rays_d.expandDims(-2).mul(z_vals)), givePerterb(getSample(t_vals, near_full, far_full).expandDims(-1)))
    //输出分别为：
    //pos：尺寸(IvH, IvW, N_samples, 3)
    //z_vals_full：尺寸(H, W, N_samples, 1)
  }

  def SH2(viewdir: NDArray): NDArray = {
    //viewdir是输入的方向视角，最高维大小为3，分别是x，y和z
    //最高维经过归一化
    val outputList = new NDList(9)

    val x = viewdir.get("...,0")
    val y = viewdir.get("...,1")
    val z = viewdir.get("...,2")

    val cosPhi = z
    val sinPhi = z.square().sub(1).neg().sqrt()
    //TODO：rsub更新以后做修改
    val cosTheta = x.div(sinPhi)
    val sinTheta = y.div(sinPhi)
    val sinThetaCosPhi = sinTheta.mul(cosPhi)
    val sinThetaSinPhi = sinTheta.mul(sinPhi)

    //l=0
    outputList.add(viewdir.getManager.ones(cosTheta.getShape, cosTheta.getDataType))
    //l=1
    outputList.add(cosTheta)
    outputList.add(sinThetaCosPhi)
    outputList.add(sinThetaSinPhi)
    //l=2
    outputList.add(cosTheta.square().sub(1.0 / 3))
    outputList.add(sinThetaCosPhi.mul(cosTheta))
    outputList.add(sinThetaSinPhi.mul(cosTheta))
    outputList.add(sinThetaCosPhi.mul(cosPhi).mul(2).sub(sinTheta).mul(sinTheta))
    //sinTheta * sinTheta * (2 * cosPhi * cosPhi - 1)
    outputList.add(sinThetaSinPhi.mul(sinTheta).mul(cosPhi))
    //sinTheta * sinTheta * sinPhi * cosPhi

    NDArrays.stack(outputList, -1)
  }

  val addNoise = if (config.raw_noise_std > 0) (input: NDArray) => input.add(input.getManager.randomNormal(input.getShape).mul(config.raw_noise_std))
  else (input: NDArray) => input

  def noise(input: Boolean): Unit = {
    if (input && config.raw_noise_std > 0) (input: NDArray) => input.add(input.getManager.randomNormal(input.getShape).mul(config.raw_noise_std))
    else (input: NDArray) => input
  }

  def render_rgb(rgb: NDArray, d: NDArray, z_vals_full: NDArray): NDArray = {
    //将网络的输出渲染成特征图
    //rgb：尺寸(H, W, N_samples, 3)
    //d：尺寸(H, W, N_samples - 1, 1)
    //z_vals_full：尺寸(H, W, N_samples, 1)
    val dist = z_vals_full.get(from1).sub(z_vals_full.get(toNeg1))
    //rays_d归一化过，dist此时已经是真实世界的距离
    var alpha = addNoise(d).getNDArrayInternal.relu().neg().mul(dist)
    val T = alpha.cumSum(1).exp()
    //最前面是1
    alpha = alpha.exp().sub(1).neg()
    val weight = alpha.get(to1).getNDArrayInternal.concat(new NDList(T.get(toNeg1).mul(alpha.get(from1)), T.get(fromNeg1)), -2)
    weight.mul(rgb.getNDArrayInternal.sigmoid()).sum(Array(-2))
    //输出：
    //rgb_out：尺寸(H, W, 3)
  }

  val ndc = if (config.ndc) (H: Int, W: Int, focal: Float, rays_o: NDArray, rays_d: NDArray, rays_d_full: NDArray) => ndc_rays(H: Int, W: Int, focal: Float, rays_o: NDArray, rays_d: NDArray, rays_d_full: NDArray)
  else (H: Int, W: Int, focal: Float, rays_o: NDArray, rays_d: NDArray, rays_d_full: NDArray) => (rays_o, rays_d, rays_d_full)

  val getNear = if (config.ndc) (near: Float, norm: NDArray) => norm.getManager.zeros(norm.getShape, norm.getDataType)
  else (near: Float, norm: NDArray) => norm.mul(near)

  val getFar = if (config.ndc) (far: Float, norm: NDArray) => norm
  else (far: Float, norm: NDArray) => norm.mul(far)

  val getD = new NDIndex("...,:-1,:1")
  val getF = new NDIndex("...,1:")

  def forward(H: Int, W: Int, focal: Float, c2w: NDArray, rays_o: NDArray, near: Float, far: Float, training: Boolean): NDArray = {
    //输入在get_rays_np中有详细介绍
    //rays_o：尺寸(3)，光线起点
    //near：近端平面到原点距离，如果启动ndc的话这个值被看做是0
    //far：远端平面到原点距离，如果启动ndc的话这个值被看做是1
    require(H % (1 << config.factor) == 0 && W % (1 << config.factor) == 0)
    val IvH = H >> config.factor
    val IvW = W >> config.factor
    val (rays_d, rays_d_full) = get_rays_np(H, W, focal, IvH, IvW, c2w)
    val (rays_o2, rays_d2, rays_d_full2) = ndc(H, W, focal, rays_o.reshape(1, 1, 3), rays_d, rays_d_full)
    val norm = rays_d2.norm(Array(-1), true)
    val rays_d3 = rays_d2.div(norm)
    val near2 = getNear(near, norm)
    val far2 = getFar(far, norm)
    val norm_full = rays_d_full2.norm(Array(-1), true)
    val rays_d_full3 = rays_d_full2.div(norm_full)
    val near_full2 = getNear(near, norm_full)
    val far_full2 = getFar(far, norm_full)
    val (pos, z_vals_full) = getInput(rays_o2, rays_d3, near2, far2, near_full2, far_full2)
    val fd = config.block.forward(new NDList(positionCode(pos, config.pos_L)), training).get(0)
    val d = fd.get(getD)
    val f = fd.get(getF).reshape(H, W, config.N_samples, 3, 9)
    val sh = SH2(rays_d_full3).reshape(H, W, 1, 1, 9)
    val rgb = f.mul(sh).sum(Array(-1))
    render_rgb(rgb, d, z_vals_full)
    //输出：
    //rgb_out：渲染结果，尺寸(H, W, 3)
  }
}