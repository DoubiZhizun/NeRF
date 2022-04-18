package nerf_change

import ai.djl.ndarray._
import ai.djl.nn._
import ArrayFnc._
import ai.djl.ndarray.index._
import ai.djl.ndarray.types.DataType

import java.util.function._
import scala.collection.JavaConverters._

class nerf(config: nerfConfig) {

  val block = new SequentialBlock()

  val posPosCode = if (config.pos_L > 0) positionCode(_: NDArray, config.pos_L) else (x: NDArray) => x
  val dirPosCode = if (config.direction_L > 0) positionCode(_: NDArray, config.direction_L) else (x: NDArray) => x
  val posCode = new Function[NDList, NDList] {
    override def apply(input: NDList): NDList = new NDList(posPosCode(input.get(0)), dirPosCode(input.get(1)))
  }
  block.add(new Function[NDList, NDList] {
    override def apply(input: NDList): NDList = getInput(input)
  }).add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
    override def apply(input: java.util.List[NDList]): NDList = new NDList(input.get(0).get(0), input.get(0).get(1), input.get(1).get(1), input.get(1).get(0), input.get(1).get(2), input.get(1).get(3))
  }, List[Block](new SequentialBlock().add(posCode).add(config.coarseBlock), new LambdaBlock(new Function[NDList, NDList] {
    override def apply(input: NDList): NDList = new NDList(input.get(1), input.get(2), input.get(3), input.get(4))
  })).asJava)).add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
    override def apply(input: java.util.List[NDList]): NDList = new NDList(input.get(0).get(0), input.get(0).get(1), input.get(1).get(0), input.get(1).get(1), input.get(1).get(2), input.get(1).get(3))
  }, List[Block](new LambdaBlock(new Function[NDList, NDList] {
    override def apply(input: NDList): NDList = render_ray(input)
  }), new LambdaBlock(new Function[NDList, NDList] {
    override def apply(input: NDList): NDList = new NDList(input.get(3), input.get(2), input.get(4), input.get(5))
  })).asJava)).add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
    override def apply(input: java.util.List[NDList]): NDList = new NDList(input.get(0).singletonOrThrow().concat(input.get(1).get(0), -1))
  }, List[Block](new LambdaBlock(new Function[NDList, NDList] {
    override def apply(input: NDList): NDList = new NDList(input.get(0))
  }), new SequentialBlock().add(new Function[NDList, NDList] {
    override def apply(input: NDList): NDList = sample_pdf(input)
  }).add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
    override def apply(input: java.util.List[NDList]): NDList = new NDList(input.get(0).get(0), input.get(0).get(1), input.get(1).get(0))
  }, List[Block](new SequentialBlock().add(posCode).add(config.fineBlock), new LambdaBlock(new Function[NDList, NDList] {
    override def apply(input: NDList): NDList = new NDList(input.get(2))
  })).asJava)).add(new Function[NDList, NDList] {
    override def apply(input: NDList): NDList = render_ray(input)
  })).asJava)) //输出最高维0到3是粗糙网络输出，3到6是细腻网络输出


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
    rgb.add(NDArrays.sub(1, weight.sum(Array(1))))
    //对于这个函数的理解：
    //假设渲染出的所有点的颜色都是白色，计算最终渲染出的颜色，
    //然后用纯白色去减这个颜色，就能得到背景所需的颜色
  } else (rgb: NDArray, weight: NDArray) => rgb

  val from1 = new NDIndex(":,1:,:")
  val to1 = new NDIndex(":,:1,:")
  val fromNeg1 = new NDIndex(":,-1:,:")
  val toNeg1 = new NDIndex(":,:-1,:")

  def render_ray(input: NDList): NDList = {
    //将网络的输出渲染成色点
    //将对于渲染所使用的参数的要求写在这里
    //渲染所输入的direction要求是归一化过的（即范数为1）
    //渲染点的计算方式 o + td，t的上下限分别是near 和 far
    //input内有三项，分别是颜色、密度和所取的点（满足上述要求）
    //前两项共有三维，第一维是batch，第二维是ray（N_samples或N_importance），第三维是对应的数据
    //颜色的第二维比密度的第二维大1（重要）
    //第三项也有三维，第一维是batch，第二维是取到的点，跟颜色的第二维长度相同，第三维长度为1
    //输出有两项，分别是：
    //每个Batch中的光线渲染出的颜色，两维，第一维是batch，第二维是颜色
    //权重，用于让粗糙网络估计精密网络的值，第一维是batch，第二维是权重，第三维是1
    val z_vals = input.get(2)
    val dist = z_vals.get(from1).sub(z_vals.get(toNeg1))
    //d归一化过，dist此时已经是真实世界的距离
    val rgb = input.get(0).getNDArrayInternal.sigmoid()
    //这里是否需要考虑优化一下？
    var alpha = addNoise(input.get(1)).getNDArrayInternal.relu().neg().mul(dist)
    val T = alpha.cumSum(1).exp()
    //最前面是1
    alpha = alpha.exp().sub(1).neg()
    val weight = alpha.get(to1).concat(T.get(toNeg1).mul(alpha.get(from1)), 1)
    val weight2 = weight.concat(T.get(fromNeg1), 1)
    new NDList(addBkgd(weight2.mul(rgb).sum(Array(1)), weight2), weight)
  }

  val getSample = if (config.lindisp) (t_vals: NDArray, near: NDArray, far: NDArray) => NDArrays.div(1, NDArrays.div(1, near).mul(NDArrays.sub(1, t_vals)).add(NDArrays.div(1, far).mul(t_vals)))
  else (t_vals: NDArray, near: NDArray, far: NDArray) => near.mul(t_vals.sub(1).neg()).add(far.mul(t_vals))

  val givePerterb = if (config.perterb) (z_vals: NDArray) => {
    val manager = z_vals.getManager
    val mids = z_vals.get(from1).add(z_vals.get(toNeg1)).mul(.5)
    val upper = mids.concat(z_vals.get(fromNeg1), 1)
    val lower = z_vals.get(to1).concat(mids, 1)
    val t_rand = manager.randomUniform(0, 1, z_vals.getShape)
    lower.add(upper.sub(lower).mul(t_rand))
  } else (z_vals: NDArray) => z_vals

  val to1NoTail = new NDIndex(":,:1")
  val from1NoTail = new NDIndex(":,1:")

  def getInput(input: NDList): NDList = {
    //为网络准备输入
    //input有四个元素
    //分别是光线起点，光线方向，bound和观察方向
    //光线方向，观察方向和选中的点都应该是归一化过的
    //光线起点和方向都是二维
    //bound是二维，最高维度分别是near，far
    //观察方向是二维
    val rays_o = input.get(0).expandDims(1)
    val rays_d = input.get(1).expandDims(1)
    //TODO: add ndc here
    val bound = input.get(2)
    val near = bound.get(to1NoTail)
    val far = bound.get(from1NoTail)
    val viewdirs = input.get(3)
    val manager = rays_o.getManager

    val t_vals = manager.linspace(0, 1, config.N_samples)
    val z_vals = givePerterb(getSample(t_vals, near, far).expandDims(2))
    new NDList(rays_o.add(rays_d.mul(z_vals)), viewdirs, z_vals, rays_o, rays_d)
  }

  val idxArray = (0 until config.N_importance).map { i =>
    new NDIndex(s":,$i:${i + 1}")
  }

  def sample_pdf(input: NDList): NDList = {
    //input.get(1)是weight
    //input.get(2)是viewdirs
    //input.get(3)是z_vals
    //input.get(4)是rays_o
    //input.get(5)是rays_d

    val weight = input.get(1).stopGradient().add(1e-5) //防止0，同时停止追踪梯度
    val manager = weight.getManager
    val weightCum = weight.cumSum(1)
    val cdf = weightCum.div(weightCum.get(fromNeg1))
    //长为N_sample-1

    val viewdirs = input.get(2)
    val z_vals = input.get(3)
    val rays_o = input.get(4)
    val rays_d = input.get(5)

    val samples = manager.create(samplePdf(cdf.toType(DataType.FLOAT32, false).toFloatArray, z_vals.toType(DataType.FLOAT32, false).toFloatArray, cdf.getShape.get(0).toInt, config.N_samples, config.N_importance)).reshape(cdf.getShape.get(0), config.N_samples + config.N_importance, 1)

    new NDList(rays_o.add(rays_d.mul(samples)), viewdirs, samples)
    //这个函数后续需要更多修改
  }

  def getBlock(): Block = block
  //该模块四个输入分别是光线起点，光线方向，bound和观察方向
  //一个输出为两个网络渲染出的颜色
}