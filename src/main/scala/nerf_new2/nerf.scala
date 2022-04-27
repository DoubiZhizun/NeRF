package nerf_new2

import ai.djl.engine._
import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.training._
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


  def get_rays_np(H: Int, W: Int, focal: Float, IvH: Int, IvW: Int, c2w: NDArray): NDArray = {
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
    dirs.matMul(c2w.transpose(Array(1, 0): _*))
    //返回：rays_d
    //光线的方向，尺寸(IvH, IvW, 3)
  }

  def ndc_rays(H: Int, W: Int, focal: Float, rays_o: NDArray, rays_d: NDArray): (NDArray, NDArray) = {
    //对光线做ndc变换
    //近端平面恒定认为是-1
    //H、W、focal和rays_d同上
    //rays_o：光线的原点，尺寸(1, 1, 3)
    val manager = rays_o.getManager
    val rays_d2 = rays_d.get(new NDIndex().addAllDim(2).addIndices(2))
    val t = rays_o.get(new NDIndex().addAllDim(2).addIndices(2)).add(1).div(rays_d2)
    val rays_d0 = rays_d.get(new NDIndex().addAllDim(2).addIndices(0))
    val rays_d1 = rays_d.get(new NDIndex().addAllDim(2).addIndices(1))
    val rays_o0 = rays_o.get(new NDIndex().addAllDim(2).addIndices(0)).sub(t.mul(rays_d0))
    val rays_o1 = rays_o.get(new NDIndex().addAllDim(2).addIndices(1)).sub(t.mul(rays_d1))
    //将所有光线的原点映射到z == -1（即近端平面）上

    val o0 = rays_o0.mul(1 / (W / (2 * focal)))
    val o1 = rays_o1.mul(1 / (H / (2 * focal)))
    val o2 = manager.full(o0.getShape, -1, rays_o.getDataType)

    val d0 = rays_d0.div(rays_d2).add(rays_o0).mul(-1 / (W / (2 * focal)))
    val d1 = rays_d1.div(rays_d2).add(rays_o1).mul(-1 / (H / (2 * focal)))
    val d2 = manager.full(o0.getShape, 2, rays_o.getDataType)

    (o0.getNDArrayInternal.stack(new NDList(o1, o2), -1), d0.getNDArrayInternal.stack(new NDList(d1, d2), -1))
    //返回：
    //rays_o：尺寸(IvH, IvW, 3)
    //rays_d：尺寸(IvH, IvW, 3)
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

  def getInput(rays_o: NDArray, rays_d: NDArray, near: NDArray, far: NDArray): (NDArray, NDArray) = {
    //为网络准备输入
    //rays_o：尺寸(IvH, IvW, 3)或(1, 1, 3)
    //rays_d：尺寸(IvH, IvW, 3)
    //near：尺寸(IvH, IvW, 1)
    //far：尺寸(IvH, IvW, 1)
    val manager = rays_o.getManager
    val t_vals = manager.linspace(0, 1, config.N_samples)
    val z_vals = givePerterb(getSample(t_vals, near, far).expandDims(-1))
    (rays_o.expandDims(-2).add(rays_d.expandDims(-2).mul(z_vals)), z_vals)
    //输出分别为：
    //pos：尺寸(IvH, IvW, N_samples, 3)
    //z_vals：尺寸(IvH, IvW, N_samples, 1)
  }

  val addNoise = if (config.raw_noise_std > 0) (input: NDArray) => input.add(input.getManager.randomNormal(input.getShape).mul(config.raw_noise_std))
  else (input: NDArray) => input

  def noise(input: Boolean): Unit = {
    if (input && config.raw_noise_std > 0) (input: NDArray) => input.add(input.getManager.randomNormal(input.getShape).mul(config.raw_noise_std))
    else (input: NDArray) => input
  }

  def render_feature(f: NDArray, d: NDArray, z_vals: NDArray): NDArray = {
    //将网络的输出渲染成特征图
    //f：尺寸(IvH, IvW, N_samples, Mf)
    //d：尺寸(IvH, IvW, N_samples, 1)
    //z_vals：尺寸(IvH, IvW, N_samples, 1)
    val dist = z_vals.get(from1).sub(z_vals.get(toNeg1))
    //rays_d归一化过，dist此时已经是真实世界的距离
    var alpha = addNoise(d).getNDArrayInternal.relu().neg().mul(dist)
    val T = alpha.cumSum(1).exp()
    //最前面是1
    alpha = alpha.exp().sub(1).neg()
    val weight = alpha.get(to1).getNDArrayInternal.concat(new NDList(T.get(toNeg1).mul(alpha.get(from1)), T.get(fromNeg1)), -2)
    weight.mul(f.getNDArrayInternal.sigmoid()).sum(Array(-2))
    //输出：
    //f_out：尺寸(IvH, IvW, Mf)
  }

  val ndc = if (config.ndc) (H: Int, W: Int, focal: Float, rays_o: NDArray, rays_d: NDArray) => ndc_rays(H: Int, W: Int, focal: Float, rays_o: NDArray, rays_d: NDArray)
  else (H: Int, W: Int, focal: Float, rays_o: NDArray, rays_d: NDArray) => (rays_o, rays_d)

  val getNear = if (config.ndc) (near: Float, norm: NDArray) => norm.getManager.zeros(norm.getShape, norm.getDataType)
  else (near: Float, norm: NDArray) => norm.mul(near)

  val getFar = if (config.ndc) (far: Float, norm: NDArray) => norm
  else (far: Float, norm: NDArray) => norm.mul(far)

  def forward(H: Int, W: Int, focal: Float, c2w: NDArray, rays_o: NDArray, near: Float, far: Float, training: Boolean): NDArray = {
    //输入在get_rays_np中有详细介绍
    //rays_o：尺寸(3)，光线起点
    //near：近端平面到原点距离，如果启动ndc的话这个值被看做是0
    //far：远端平面到原点距离，如果启动ndc的话这个值被看做是1
    require(H % (1 << config.factor) == 0 && W % (1 << config.factor) == 0)
    val IvH = H >> config.factor
    val IvW = W >> config.factor
    val rays_d = get_rays_np(H, W, focal, IvH, IvW, c2w)
    val (rays_o2, rays_d2) = ndc(H, W, focal, rays_o.reshape(1, 1, 3), rays_d)
    val norm = rays_d2.norm(Array(-1), true)
    val rays_d3 = rays_d2.div(norm)
    val near2 = getNear(near, norm)
    val far2 = getFar(far, norm)
    val (pos, z_vals) = getInput(rays_o2, rays_d3, near2, far2)
    val fd = config.mlpBlock.forward(new NDList(positionCode(pos, config.pos_L), positionCode(rays_d, config.dir_L)), training)
    val f_out = render_feature(fd.get(0), fd.get(1), z_vals)
    config.cnnBlock.forward(new NDList(f_out.transpose(Array(2, 0, 1): _*).expandDims(0)), training).singletonOrThrow().squeeze().transpose(Array(1, 2, 0): _*)
    //输出：
    //rgb_out：渲染结果，尺寸(H, W, 3)
  }
}