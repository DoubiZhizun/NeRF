package nerf

import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.nn.core._

import java.io._
import java.util.function._
import scala.collection.JavaConverters._
import scala.collection.mutable._

class coreBlock(config: nerfConfig) {
  //核心模块

  var mlpBlock = new SequentialBlock()

  for (i <- 0 until config.D) {
    mlpBlock.add(Linear.builder().setUnits(config.W).build()).add(Activation.reluBlock())
    if (config.skips.contains(i)) {
      mlpBlock = new SequentialBlock().add(new ParallelBlock(new Function[java.util.List[NDList], NDList] {
        override def apply(x: java.util.List[NDList]): NDList = new NDList(x.get(0).singletonOrThrow().concat(x.get(1).singletonOrThrow(), -1))
      }, List[Block](mlpBlock, new LambdaBlock(new Function[NDList, NDList] {
        override def apply(t: NDList): NDList = t
      })).asJava))
    }
  }

  val inputFnc = if (config.useTime && config.timeL == 0) (pos: NDArray, t: NDArray) => pos.concat(t, -1)
  else (pos: NDArray, t: NDArray) => pos
  //输入函数，如果需要使用时间且时间是输入神经网络的话则将pos和t拼接，否则只输出pos

  mlpBlock = new SequentialBlock().add(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(positionCode(inputFnc(t.get(0), t.get(1)), config.posL))
  }).add(mlpBlock)

  //全连接网络构造完毕
  //该模块输入为：点的位置和时间
  //该模块输出为：W个点的特征

  val densityWeight = Parameter.builder().setType(Parameter.Type.WEIGHT).optShape(new Shape(1, config.W)).build()
  val densityBias = Parameter.builder().setType(Parameter.Type.BIAS).optShape(new Shape(1)).build()
  //得到密度

  var rgbBlock: Block = null

  val timeInputSize = 1 + 2 * config.timeL
  val shOutputSize = if (config.useTime) 3 * timeInputSize else 3
  val finalOutputSize = if (config.useSH) shOutputSize * 9 else shOutputSize

  //先用球谐函数系数拟合不同方向的颜色的傅里叶系数
  //再用傅里叶系数拟合不同时间的颜色

  if (config.useSH) {
    rgbBlock = new ParallelBlock(new Function[java.util.List[NDList], NDList] {
      override def apply(t: java.util.List[NDList]): NDList = new NDList(t.get(0).singletonOrThrow().mul(t.get(1).singletonOrThrow()).sum(Array(-1)))
    }, List[Block](new SequentialBlock().add(new LambdaBlock(new Function[NDList, NDList] {
      override def apply(t: NDList): NDList = new NDList(t.get(0))
    })).add(Linear.builder().setUnits(finalOutputSize).build()).add(new LambdaBlock(new Function[NDList, NDList] {
      override def apply(t: NDList): NDList = {
        val elem = t.singletonOrThrow()
        val shape = Shape.update(elem.getShape, elem.getShape.dimension() - 1, shOutputSize).add(9)
        new NDList(elem.reshape(shape))
      }
    })), new LambdaBlock(new Function[NDList, NDList] {
      override def apply(t: NDList): NDList = new NDList(SH2(t.get(1)).expandDims(-2))
    })).asJava)
  } else {
    rgbBlock = new SequentialBlock().add(new LambdaBlock(new Function[NDList, NDList] {
      override def apply(t: NDList): NDList = new NDList(t.get(0).concat(positionCode(t.get(1), config.dirL), -1))
    })).add(Linear.builder().setUnits(config.W / 2).build()).add(Activation.reluBlock()).add(Linear.builder().setUnits(finalOutputSize).build())
  }

  if (config.useTime && config.timeL != 0) {
    rgbBlock = new ParallelBlock(new Function[java.util.List[NDList], NDList] {
      override def apply(t: java.util.List[NDList]): NDList = new NDList(t.get(0).singletonOrThrow().mul(t.get(1).singletonOrThrow()).sum(Array(-1)))
    }, List[Block]((if (config.useSH) new SequentialBlock().add(rgbBlock) else rgbBlock.asInstanceOf[SequentialBlock]).add(new LambdaBlock(new Function[NDList, NDList] {
      override def apply(t: NDList): NDList = {
        val elem = t.singletonOrThrow()
        val shape = Shape.update(elem.getShape, elem.getShape.dimension() - 1, 3).add(timeInputSize)
        new NDList(elem.reshape(shape))
      }
    })), new LambdaBlock(new Function[NDList, NDList] {
      override def apply(t: NDList): NDList = new NDList(Fourier(t.get(2)).expandDims(-2))
    })).asJava)
  }

  //颜色获取网络初始化完毕
  //该模块输入为：全连接网络输出的特征，方向和时间
  //该模块输出为：rgb

  def positionCode(input: NDArray, L: Int): NDArray = {
    //sin cos位置编码，L为编码阶数
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

  val freq = (1 to config.timeL).map(i => 2 * i * Math.PI)

  def Fourier(t: NDArray): NDArray = {
    //L阶傅里叶级数
    //t的范围是0到1
    val outputList = new NDList(config.timeL * 2 + 1)
    outputList.add(t.onesLike())
    for (f <- freq) {
      val tpi = t.mul(f)
      outputList.add(tpi.cos())
      outputList.add(tpi.sin())
    }
    NDArrays.stack(outputList, -1)
  }

  def SH2(viewdir: NDArray): NDArray = {
    //二阶球谐函数
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
    outputList.add(x.onesLike())
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

  def forward(pos: NDArray, dir: NDArray, time: NDArray, training: Boolean): (NDArray, NDArray) = {
    //输入：
    //pos：位置，最高维度尺寸为3，代表x、y、z，范围-1到1
    //dir：方向，最高维度尺寸为3，代表x、y、z，范围-1到1
    //time：时间，最高维度尺寸为1，代表t，范围0到1，没有则为null
    //training：如果在训练则拉高
    val mlpBlockOutput = mlpBlock.forward(config.ps, new NDList(pos, time), training).singletonOrThrow()
    val density = mlpBlockOutput.getNDArrayInternal.linear(mlpBlockOutput, config.ps.getValue(densityWeight, config.device, training), config.ps.getValue(densityBias, config.device, training)).singletonOrThrow().getNDArrayInternal.relu()
    val rgb = rgbBlock.forward(config.ps, new NDList(mlpBlockOutput, dir, time), training).singletonOrThrow()
    (density, rgb)
    //返回：
    //density：密度，最高维度尺寸为1，代表密度，大于0
    //rgb：颜色，最高维度尺寸为3，代表r、g、b，范围0到1
  }

  def initialize(manager: NDManager): coreBlock = {
    mlpBlock.initialize(manager, DataType.FLOAT32, new Shape(1, 1, 3), new Shape(1, 1, 1))
    densityWeight.initialize(manager, DataType.FLOAT32)
    densityBias.initialize(manager, DataType.FLOAT32)
    rgbBlock.initialize(manager, DataType.FLOAT32, new Shape(1, 1, config.W), new Shape(1, 1, 3), new Shape(1, 1, 1))
    this
  }

  def save(os: DataOutputStream): Unit = {
    mlpBlock.saveParameters(os)
    densityWeight.save(os)
    densityBias.save(os)
    rgbBlock.saveParameters(os)
  }

  def load(manager: NDManager, is: DataInputStream): Unit = {
    mlpBlock.loadParameters(manager, is)
    densityWeight.load(manager, is)
    densityBias.load(manager, is)
    rgbBlock.loadParameters(manager, is)
  }
}