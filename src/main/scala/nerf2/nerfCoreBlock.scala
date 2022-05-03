package nerf2

import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.nn.core._
import ai.djl.training._
import ai.djl.util._
import nerf2.nerfCoreBlock._

import java.util
import java.util.function.Function
import scala.collection.JavaConverters._

final class nerfCoreBlock(config: nerfConfig, isCoarse: Boolean) extends AbstractBlock(VERSION) {

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
          .add(Linear.builder().setUnits(1).build()),
        new LambdaBlock(toFunction((t: NDList) => t))
      ).asJava
    )
  )

  //mlpBlock是全连接模块，输入为经过位置编码的post，输出为d和W维特征

  addChildBlock(mlpBlock.getClass.getSimpleName, mlpBlock)

  private var getRgbBlock: Block = null

  if (config.useDir) {
    getRgbBlock = new SequentialBlock().add(
      new ParallelBlock(
        toFunction((t: util.List[NDList]) => new NDList(t.get(0).singletonOrThrow().add(t.get(1).singletonOrThrow()))),
        List[Block](
          new SequentialBlock()
            .add(toFunction((t: NDList) => new NDList(t.get(0))))
            .add(Linear.builder().setUnits(config.W / 2).build()),
          new SequentialBlock()
            .add(toFunction((t: NDList) => new NDList(t.get(1))))
            .add(Linear.builder().setUnits(config.W / 2).build())
        ).asJava
      )
    ).add(Linear.builder().setUnits(3).build())
    //输入为256维特征和经过位置编码的dir，输出为rgb
  } else {
    getRgbBlock = Linear.builder().setUnits(3).build()
    //输入为256维特征，输出为rgb
  }

  addChildBlock(getRgbBlock.getClass.getSimpleName, getRgbBlock)

  private val inputFunction = if (config.useDir) (f: NDArray, dir: NDArray) => new NDList(f, positionCode(dir, config.dirL))
  else (f: NDArray, dir: NDArray) => new NDList(f)
  //getRgbBlock的输入函数

  override def initializeChildBlocks(manager: NDManager, dataType: DataType, inputShapes: Shape*): Unit = {
    val postShape = inputShapes(0)
    mlpBlock.initialize(manager, dataType, Shape.update(postShape, postShape.dimension() - 1, postShape.tail() * (1 + 2 * config.postL)))
    if (config.useDir) {
      val dirShape = inputShapes(1)
      getRgbBlock.initialize(manager, dataType, Shape.update(postShape, postShape.dimension() - 1, config.W), Shape.update(dirShape, dirShape.dimension() - 1, dirShape.tail() * (1 + 2 * config.dirL)))
    } else {
      getRgbBlock.initialize(manager, dataType, Shape.update(postShape, postShape.dimension() - 1, config.W))
    }
  }

  override def forwardInternal(parameterStore: ParameterStore, inputs: NDList, training: Boolean, params: PairList[String, AnyRef]): NDList = {
    val mlpOutput = mlpBlock.forward(parameterStore, new NDList(positionCode(inputs.get(0), config.postL)), training, params)
    val rgbOutput = if (isCoarse && !training) null else getRgbBlock.forward(parameterStore, inputFunction(mlpOutput.get(1), inputs.get(1)), training, params).singletonOrThrow()
    new NDList(mlpOutput.get(0), rgbOutput)
    //输出d和rgb
  }

  override def getOutputShapes(inputShapes: Array[Shape]): Array[Shape] = {
    val postShape = inputShapes(0)
    Array(Shape.update(postShape, postShape.dimension() - 1, 1), Shape.update(postShape, postShape.dimension() - 1, 3))
  }
}

object nerfCoreBlock {
  private val VERSION: Byte = 0

  private def toFunction[T1, T2](input: T1 => T2): Function[T1, T2] = new Function[T1, T2] {
    override def apply(t: T1): T2 = input(t)
  }

  private def positionCode(input: NDArray, L: Int): NDArray = {
    //sin cos位置编码，L为编码阶数
    val output = new NDList(L * 2)
    for (i <- 0 until L) {
      val inputMulFactor = input.mul(Math.PI * (1 << i))
      output.add(inputMulFactor.sin())
      output.add(inputMulFactor.cos())
    }
    input.getNDArrayInternal.concat(output, -1)
  }
}