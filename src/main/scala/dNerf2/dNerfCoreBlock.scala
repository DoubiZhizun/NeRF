package dNerf2

import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.nn.core._
import ai.djl.training._
import ai.djl.util._
import dNerf2.dNerfCoreBlock._

import java.util
import java.util.function.Function
import scala.collection.JavaConverters._

final class dNerfCoreBlock(config: dNerfConfig, isCoarse: Boolean) extends AbstractBlock(VERSION) {

  private var mlpBlock = new SequentialBlock()

  for (i <- 0 until config.D) {
    mlpBlock.add(Linear.builder().setUnits(config.W).build()).add(Activation.reluBlock())
    if (config.skips.contains(i)) {
      mlpBlock = new SequentialBlock().add(
        new ParallelBlock(
          toFunction((t: util.List[NDList]) => new NDList(t.get(0).singletonOrThrow().concat(t.get(1).singletonOrThrow(), -1))),
          List[Block](
            mlpBlock,
            new LambdaBlock(toFunction((t: NDList) => t))
          ).asJava
        )
      )
    }
  }

  mlpBlock.add(
    new ParallelBlock(
      toFunction((t: util.List[NDList]) => new NDList(t.get(0).singletonOrThrow(), t.get(1).singletonOrThrow())),
      List[Block](
        new SequentialBlock()
          .add(toFunction((t: NDList) => new NDList(t.singletonOrThrow().get(":-1"))))
          .add(Linear.builder().setUnits(if (config.useTime) 1 + config.fourierL * 2 else 1).build()),
        new LambdaBlock(toFunction((t: NDList) => t))
      ).asJava
    )
  )

  //mlpBlock是全连接模块，输入为经过位置编码的pos，输出为d和W维特征

  addChildBlock(mlpBlock.getClass.getSimpleName, mlpBlock)

  private var getRgbBlock: Block = null
  private val outputSize = if (config.useTime) 3 * (1 + 2 * config.fourierL) else 3


  if (config.useDir) {
    if (config.useSH) {
      getRgbBlock = new ParallelBlock(
        toFunction((t: util.List[NDList]) => new NDList(t.get(0).singletonOrThrow().reshape(Shape.update(t.get(0).get(0).getShape, t.get(0).get(0).getShape.dimension() - 1, 9).add(outputSize)).mul(t.get(1).get(1).expandDims(-1)).sum(Array(-2)))),
        List[Block](
          new SequentialBlock()
            .add(toFunction((t: NDList) => new NDList(t.get(0))))
            .add(Linear.builder().setUnits(outputSize * 9).build()),
          new LambdaBlock(toFunction((t: NDList) => t))
        ).asJava
      )
      //输入为W维特征和计算过球谐函数的dir，输出为rgb
    } else {
      getRgbBlock = new SequentialBlock().add(
        new ParallelBlock(
          toFunction((t: util.List[NDList]) => new NDList(t.get(0).singletonOrThrow().add(t.get(1).singletonOrThrow()).getNDArrayInternal.relu())),
          List[Block](
            new SequentialBlock()
              .add(toFunction((t: NDList) => new NDList(t.get(0))))
              .add(Linear.builder().setUnits(config.W / 2).build()),
            new SequentialBlock()
              .add(toFunction((t: NDList) => new NDList(t.get(1))))
              .add(Linear.builder().setUnits(config.W / 2).build())
          ).asJava
        )
      ).add(Linear.builder().setUnits(outputSize).build())
      //输入为W维特征和经过位置编码的dir，输出为rgb
    }
  } else {
    getRgbBlock = Linear.builder().setUnits(outputSize).build()
    //输入为W维特征，输出为rgb
  }

  addChildBlock(getRgbBlock.getClass.getSimpleName, getRgbBlock)

  private val inputFunction = if (config.useDir) if (config.useSH) (f: NDArray, dir: NDArray) => new NDList(f, SH2(dir))
  else (f: NDArray, dir: NDArray) => new NDList(f, positionCode(dir, config.dirL))
  else (f: NDArray, dir: NDArray) => new NDList(f)
  //getRgbBlock的输入函数

  private val alphaOutput = if (config.useTime) (alpha: NDArray, times: NDArray) => {
    val timesFourier = fourier(times, config.fourierL)
    new NDList(alpha.mul(timesFourier).sum(Array(-1), true), timesFourier)
  } else (alpha: NDArray, times: NDArray) => new NDList(alpha, times)
  //alpha输出函数

  private val rgbOutput = if (config.useTime) (rgb: NDArray, fourierTimes: NDArray) => new NDList(rgb.reshape(Shape.update(rgb.getShape, rgb.getShape.dimension() - 1, 1 + 2 * config.fourierL).add(3)).mul(fourierTimes.expandDims(-1)).sum(Array(-2)))
  else (rgb: NDArray, fourierTimes: NDArray) => new NDList(rgb)
  //rgb输出函数

  override def initializeChildBlocks(manager: NDManager, dataType: DataType, inputShapes: Shape*): Unit = {
    val postShape = inputShapes(0)
    mlpBlock.initialize(manager, dataType, Shape.update(postShape, postShape.dimension() - 1, postShape.tail() * (1 + 2 * config.posL)))
    if (config.useDir) {
      val dirShape = inputShapes(1)
      if (config.useSH) {
        getRgbBlock.initialize(manager, dataType, Shape.update(postShape, postShape.dimension() - 1, config.W), Shape.update(dirShape, dirShape.dimension() - 1, 27))
      } else {
        getRgbBlock.initialize(manager, dataType, Shape.update(postShape, postShape.dimension() - 1, config.W), Shape.update(dirShape, dirShape.dimension() - 1, dirShape.tail() * (1 + 2 * config.dirL)))
      }
    } else {
      getRgbBlock.initialize(manager, dataType, Shape.update(postShape, postShape.dimension() - 1, config.W))
    }
  }

  override def forwardInternal(parameterStore: ParameterStore, inputs: NDList, training: Boolean, params: PairList[String, AnyRef]): NDList = {
    val mlpBlockOutput = mlpBlock.forward(parameterStore, new NDList(positionCode(inputs.get(0), config.posL)), training, params)
    val alphaTimes = alphaOutput(mlpBlockOutput.get(0), inputs.get(2))
    val rgbBlockOutput = if (isCoarse && !training) null else getRgbBlock.forward(parameterStore, inputFunction(mlpBlockOutput.get(1), inputs.get(1)), training, params).singletonOrThrow()
    val rgb = rgbOutput(rgbBlockOutput, alphaTimes.get(1)).singletonOrThrow()
    new NDList(alphaTimes.get(0), rgb)
    //输出d和rgb
  }

  override def getOutputShapes(inputShapes: Array[Shape]): Array[Shape] = {
    val postShape = inputShapes(0)
    Array(Shape.update(postShape, postShape.dimension() - 1, 1), Shape.update(postShape, postShape.dimension() - 1, 3))
  }
}

object dNerfCoreBlock {
  private val VERSION: Byte = 0

  private def toFunction[T1, T2](input: T1 => T2): Function[T1, T2] = new Function[T1, T2] {
    override def apply(t: T1): T2 = input(t)
  }

  private def positionCode(input: NDArray, L: Int): NDArray = {
    //sin cos位置编码，L为编码阶数
    val output = new NDList(L * 2)
    for (i <- 0 until L) {
      val inputMulFactor = input.mul(1 << i) //原文貌似就没有PI？
      output.add(inputMulFactor.sin())
      output.add(inputMulFactor.cos())
    }
    input.getNDArrayInternal.concat(output, -1)
  }

  private def SH2(viewdir: NDArray): NDArray = {
    //二阶球谐函数
    //viewdir是输入的方向视角，最高维大小为3，分别是x，y和z
    //最高维经过归一化
    val outputList = new NDList(9)

    val x = viewdir.get("...,0")
    val y = viewdir.get("...,1")
    val z = viewdir.get("...,2")

    val cosPhi = z
    val sinPhi = z.square().sub(1).neg().sqrt()
    val ones = x.onesLike()
    val zeros = y.zerosLike()
    //TODO：rsub更新以后做修改
    val cosTheta = NDArrays.where(sinPhi.lt(1e-5), ones, x.div(sinPhi))
    val sinTheta = NDArrays.where(sinPhi.lt(1e-5), zeros, y.div(sinPhi))
    val sinThetaCosPhi = sinTheta.mul(cosPhi)
    val sinThetaSinPhi = sinTheta.mul(sinPhi)

    //l=0
    outputList.add(ones)
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

  private def fourier(times: NDArray, L: Int): NDArray = {
    //傅里叶函数
    //times：0到1
    val outputList = new NDList(1 + 2 * L)
    outputList.add(times.onesLike())
    for (i <- 0 until L) {
      val times2PiN = times.mul((i + 1) * Math.PI * 2)
      outputList.add(times2PiN.sin())
      outputList.add(times2PiN.cos())
    }
    NDArrays.stack(outputList, -1)
  }
}