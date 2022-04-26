package nerf_new

import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.ndarray.types._
import ai.djl.nn._

import java.util.function._
import scala.collection.JavaConverters._
import ArrayFnc._
import ai.djl.engine._
import ai.djl.training._
import ai.djl.training.loss._
import ai.djl.training.optimizer._
import ai.djl.training.tracker._

class nerf(config: nerfConfig, ps: ParameterStore) {

  //目前已加入的改进：
  //1、使用球谐函数描述观察视角对颜色的影响
  //2、

  val loss = Loss.l2Loss("L2Loss", 1)

  def predict(rays_o: NDArray, rays_d: NDArray, near: NDArray, far: NDArray, viewdir: NDArray): NDArray = {
    val (_, fineRgbOut) = forward(rays_o, rays_d, near, far, viewdir, false)
    fineRgbOut
  }

  def train(rays_o: NDArray, rays_d: NDArray, near: NDArray, far: NDArray, viewdir: NDArray, label: NDArray): Float = {
    //label：尺寸(batchNum,3)
    val collector = Engine.getInstance().newGradientCollector()
    val (coarseRgbOut, fineRgbOut) = forward(rays_o, rays_d, near, far, viewdir, true)
    val lossValue = lossCalculate(coarseRgbOut, fineRgbOut, label)
    collector.backward(lossValue)
    collector.close()
    ps.updateAllParameters()
    lossValue.getFloat()
  }

  val forward = if (config.coarseBlock == null) forwardNoCoarse _ else forwardWithCoarse _
  val lossCalculate = if (config.coarseBlock == null) (coarse: NDArray, fine: NDArray, label: NDArray) => loss.evaluate(new NDList(label), new NDList(fine)) else (coarse: NDArray, fine: NDArray, label: NDArray) => loss.evaluate(new NDList(label), new NDList(coarse)).add(loss.evaluate(new NDList(label), new NDList(fine)))

  def forwardWithCoarse(rays_o: NDArray, rays_d: NDArray, near: NDArray, far: NDArray, viewdir: NDArray, training: Boolean): (NDArray, NDArray) = {
    //输入在getInput中有详细介绍
    //viewdir：尺寸(batchNum,1,1,3)
    val (coarsePos, coarseZ_vals) = getInput(rays_o, rays_d, near, far)
    val sh = SH2(viewdir)
    val (coarse1, coarse2, coarseD) = config.coarseBlock.forward(positionCode(coarsePos, config.pos_L), training)
    val coarseRgb = coarse2.mul(sh).sum(Array(3)).add(coarse1)
    val (coarseRgbOut, coarseWeight) = render_ray(coarseRgb, coarseD, coarseZ_vals)
    val (finePos, fineZ_vals) = sample_pdf(coarseWeight, coarseZ_vals, rays_o, rays_d)
    val (fine1, fine2, fineD) = config.fineBlock.forward(positionCode(finePos, config.pos_L), training)
    val fineRgb = fine2.mul(sh).sum(Array(3)).add(fine1)
    val (fineRgbOut, _) = render_ray(fineRgb, fineD, fineZ_vals)
    (coarseRgbOut, fineRgbOut)
    //输出：
    //coarseRgbOut：粗糙网络渲染结果，尺寸(batchNum,3)
    //fineRgbOut：细腻网络渲染结果，尺寸(batchNum,3)
  }

  def forwardNoCoarse(rays_o: NDArray, rays_d: NDArray, near: NDArray, far: NDArray, viewdir: NDArray, training: Boolean): (NDArray, NDArray) = {
    val (finePos, fineZ_vals) = getInput(rays_o, rays_d, near, far)
    val sh = SH2(viewdir)
    val (fine1, fine2, fineD) = config.fineBlock.forward(positionCode(finePos, config.pos_L), training)
    val fineRgb = fine2.mul(sh).sum(Array(3)).add(fine1)
    val (fineRgbOut, _) = render_ray(fineRgb, fineD, fineZ_vals)
    (null, fineRgbOut)
    //no coarse
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

  val addNoise = if (config.raw_noise_std > 0) (input: NDArray) => input.add(input.getManager.randomNormal(input.getShape).mul(config.raw_noise_std))
  else (input: NDArray) => input

  val addBkgd = if (config.white_bkgd) (rgb: NDArray, weight: NDArray) => {
    rgb.add(weight.sum(Array(1)).sub(1).neg())
    //对于这个函数的理解：
    //假设渲染出的所有点的颜色都是白色，计算最终渲染出的颜色，
    //然后用纯白色去减这个颜色，就能得到背景所需的颜色
  } else (rgb: NDArray, weight: NDArray) => rgb

  val from1 = new NDIndex(":,1:,:")
  val to1 = new NDIndex(":,:1,:")
  val fromNeg1 = new NDIndex(":,-1:,:")
  val toNeg1 = new NDIndex(":,:-1,:")

  def render_ray(rgb: NDArray, d: NDArray, z_vals: NDArray): (NDArray, NDArray) = {
    //将网络的输出渲染成色点
    //rgb：尺寸(batchNum,N_samples(+N_importance),3)
    //d：尺寸(batchNum,N_samples(+N_importance),1)
    //z_vals：尺寸(batchNum,N_samples(+N_importance),1)
    val dist = z_vals.get(from1).sub(z_vals.get(toNeg1))
    //rays_d归一化过，dist此时已经是真实世界的距离
    var alpha = addNoise(d).getNDArrayInternal.relu().neg().mul(dist)
    val T = alpha.cumSum(1).exp()
    //最前面是1
    alpha = alpha.exp().sub(1).neg()
    val weight = alpha.get(to1).concat(T.get(toNeg1).mul(alpha.get(from1)), 1)
    val weight2 = weight.concat(T.get(fromNeg1), 1)
    (addBkgd(weight2.mul(rgb.getNDArrayInternal.sigmoid()).sum(Array(1)), weight2), weight)
    //输出：
    //rgb_out：尺寸(batchNum,3)
    //weight：尺寸(batchNum,N_samples(+N_importance)-1)
  }

  val getSample = if (config.lindisp) (t_vals: NDArray, near: NDArray, far: NDArray) => NDArrays.div(1, NDArrays.div(1, near).mul(NDArrays.sub(1, t_vals)).add(NDArrays.div(1, far).mul(t_vals)))
  else (t_vals: NDArray, near: NDArray, far: NDArray) => near.mul(t_vals.sub(1).neg()).add(far.mul(t_vals))

  var givePerterb = if (config.perterb) (z_vals: NDArray) => {
    val manager = z_vals.getManager
    val mids = z_vals.get(from1).add(z_vals.get(toNeg1)).mul(.5)
    val upper = mids.concat(z_vals.get(fromNeg1), 1)
    val lower = z_vals.get(to1).concat(mids, 1)
    val t_rand = manager.randomUniform(0, 1, z_vals.getShape)
    lower.add(upper.sub(lower).mul(t_rand))
  } else (z_vals: NDArray) => z_vals

  def perturb(input: Boolean): Unit = {
    givePerterb = if (input && config.perterb) (z_vals: NDArray) => {
      val manager = z_vals.getManager
      val mids = z_vals.get(from1).add(z_vals.get(toNeg1)).mul(.5)
      val upper = mids.concat(z_vals.get(fromNeg1), 1)
      val lower = z_vals.get(to1).concat(mids, 1)
      val t_rand = manager.randomUniform(0, 1, z_vals.getShape)
      lower.add(upper.sub(lower).mul(t_rand))
    } else (z_vals: NDArray) => z_vals
  }

  //二阶球谐函数
  def SH2(viewdir: NDArray): NDArray = {
    //viewdir是输入的方向视角，最高维大小为3，分别是x，y和z
    //最高维经过归一化
    val outputList = new NDList(8)

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
    //l=0时为常数
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

  def getInput(rays_o: NDArray, rays_d: NDArray, near: NDArray, far: NDArray): (NDArray, NDArray) = {
    //为网络准备输入
    //rays_o：尺寸(batchNum,1,3)
    //rays_d：尺寸(batchNum,1,3)
    //near：尺寸(batchNum,1)
    //far：尺寸(batchNum,1)
    val manager = rays_o.getManager
    val t_vals = manager.linspace(0, 1, config.N_samples)
    val z_vals = givePerterb(getSample(t_vals, near, far).expandDims(2))
    (rays_o.add(rays_d.mul(z_vals)), z_vals)
    //输出分别为：
    //pos：尺寸(batchNum,N_samples,3)
    //z_vals：尺寸(batchNum,N_samples,1)
  }

  def sample_pdf(weight: NDArray, z_vals: NDArray, rays_o: NDArray, rays_d: NDArray): (NDArray, NDArray) = {
    //weight：尺寸(batchNum,N_samples-1,1)
    //z_vals：尺寸(batchNum,N_samples,1)
    //rays_o：尺寸(batchNum,1,3)
    //rays_d：尺寸(batchNum,1,3)

    val manager = weight.getManager
    val weightCum = weight.add(1e-5).stopGradient().cumSum(1)
    val cdf = weightCum.div(weightCum.get(fromNeg1))
    //长为N_sample-1

    val samples = manager.create(samplePdf(cdf.toType(DataType.FLOAT32, false).toFloatArray, z_vals.toType(DataType.FLOAT32, false).toFloatArray, cdf.getShape.get(0).toInt, config.N_samples, config.N_importance)).reshape(cdf.getShape.get(0), config.N_samples + config.N_importance, 1)

    (rays_o.add(rays_d.mul(samples)), samples)
    //pos：尺寸(batchNum,N_samples,3)
    //samples：尺寸(batchNum,N_samples,1)
  }
}