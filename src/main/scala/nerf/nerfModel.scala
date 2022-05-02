package nerf

import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.nn._
import ai.djl.nn.core._

import java.io._
import java.util
import java.util.function._
import scala.collection.JavaConverters._
import scala.collection.mutable._

class nerfModel(config: nerfConfig, isCoarse: Boolean) {
  //核心模块

  val mlpInput = if (config.useTime && !config.useFourier) new LambdaBlock(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(positionCode(t.get(0), config.posL), positionCode(t.get(1), config.timeL))
  }) else new LambdaBlock(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(positionCode(t.get(0), config.posL))
  })

  def getMlpInput(): Block = {
    if (config.useTime && !config.useFourier) {
      new ParallelBlock(new Function[util.List[NDList], NDList] {
        override def apply(t: util.List[NDList]): NDList = new NDList(t.get(0).singletonOrThrow().add(t.get(1).singletonOrThrow()))
      }, List[Block](new SequentialBlock().add(new Function[NDList, NDList] {
        override def apply(t: NDList): NDList = new NDList(t.get(0))
      }).add(Linear.builder().setUnits(config.W).build()), new SequentialBlock().add(new Function[NDList, NDList] {
        override def apply(t: NDList): NDList = new NDList(t.get(1))
      }).add(Linear.builder().setUnits(config.W).build())).asJava)
    } else {
      Linear.builder().setUnits(config.W).build()
    }
  }

  var block = new SequentialBlock().add(getMlpInput()).add(Activation.reluBlock())

  for (i <- 1 until config.D) {
    block.add(Linear.builder().setUnits(config.W).build())
    if (config.skips.contains(i - 1)) {
      block = new SequentialBlock().add(new ParallelBlock(new Function[util.List[NDList], NDList] {
        override def apply(t: util.List[NDList]): NDList = new NDList(t.get(0).singletonOrThrow().add(t.get(1).singletonOrThrow()))
      }, List[Block](block, getMlpInput()).asJava))
    }
    block.add(Activation.reluBlock())
  }

  val timeBlock = if (config.useTime && config.useFourier) new LambdaBlock(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(Fourier(t.get(1), config.fourierL))
  }) else new LambdaBlock(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(t.get(1))
  })

  block = new SequentialBlock().add(new ParallelBlock(new Function[util.List[NDList], NDList] {
    override def apply(t: util.List[NDList]): NDList = new NDList(t.get(0).singletonOrThrow(), t.get(1).singletonOrThrow())
  }, List[Block](new SequentialBlock().add(mlpInput).add(block), timeBlock).asJava))

  val bf = new Function[util.List[NDList], NDList] {
    override def apply(t: util.List[NDList]): NDList = new NDList(t.get(0).singletonOrThrow().mul(t.get(2).singletonOrThrow()), t.get(1).singletonOrThrow().sum(Array(-1), true), t.get(2).get(0))
  }

  block.add(new ParallelBlock(if (config.useTime && config.useFourier) bf else new Function[util.List[NDList], NDList] {
    override def apply(t: util.List[NDList]): NDList = new NDList(t.get(0).singletonOrThrow(), t.get(1).singletonOrThrow(), t.get(2).singletonOrThrow())
  }, List[Block](new SequentialBlock().add(new LambdaBlock(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(t.get(0))
  })).add(Linear.builder().setUnits(if (config.useTime && config.useFourier) 1 + 2 * config.fourierL else 1).build()), new LambdaBlock(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(t.get(0))
  }), new LambdaBlock(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(t.get(1))
  })).asJava))

  val dirBlock = if (config.useDir) if (config.useSH) new LambdaBlock(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(SH2(t.get(1)))
  }) else new SequentialBlock().add(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(positionCode(t.get(1), config.dirL))
  }).add(Linear.builder().setUnits(config.W / 2).build()) else new LambdaBlock(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(t.get(1))
  })

  val block2 = new SequentialBlock().add(new ParallelBlock(new Function[util.List[NDList], NDList] {
    override def apply(t: util.List[NDList]): NDList = new NDList(t.get(0).singletonOrThrow(), t.get(1).singletonOrThrow(), t.get(2).singletonOrThrow())
  }, List[Block](new SequentialBlock().add(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(t.get(0))
  }).add(Linear.builder().setUnits(config.W / 2).build()), dirBlock, new LambdaBlock(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(t.get(2))
  })).asJava))

  val processBlock = new SequentialBlock().add(if (config.useDir && !config.useSH) new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(t.get(0).add(t.get(1)).getNDArrayInternal.relu())
  } else new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = new NDList(t.get(0).getNDArrayInternal.relu())
  })

  if (config.useDir && config.useSH || config.useTime && config.useFourier) {
    val linearList: List[Block] = if (config.useDir && config.useSH) if (config.useTime && config.useFourier) List.fill(3 * (1 + 2 * config.fourierL))(Linear.builder().setUnits(9).build()) else List.fill(3)(Linear.builder().setUnits(9).build()) else List.fill(3)(Linear.builder().setUnits(1 + 2 * config.fourierL).build())
    processBlock.add(new ParallelBlock(new Function[util.List[NDList], NDList] {
      override def apply(t: util.List[NDList]): NDList = {
        val output = new NDList(t.size())
        for (i <- 0 until t.size()) {
          output.add(t.get(i).singletonOrThrow())
        }
        output
      }
    }, linearList.asJava))
  } else {
    processBlock.add(Linear.builder().setUnits(3).build())
  }

  val shF = new Function[util.List[NDList], NDList] {
    override def apply(t: util.List[NDList]): NDList = {
      val temp1 = new NDList(t.get(1).size() / 3)
      val temp2 = new NDList(t.get(1).size() / 3)
      val temp3 = new NDList(t.get(1).size() / 3)
      for (i <- 0 until(t.get(1).size(), 3)) {
        temp1.add(t.get(1).get(i).mul(t.get(0).get(1)).sum(Array(-1)))
        temp2.add(t.get(1).get(i + 1).mul(t.get(0).get(1)).sum(Array(-1)))
        temp3.add(t.get(1).get(i + 2).mul(t.get(0).get(1)).sum(Array(-1)))
      }
      new NDList(NDArrays.stack(new NDList(NDArrays.stack(temp1, -1).mul(t.get(0).get(2)).sum(Array(-1)), NDArrays.stack(temp2, -1).mul(t.get(0).get(2)).sum(Array(-1)), NDArrays.stack(temp3, -1).mul(t.get(0).get(2)).sum(Array(-1))), -1))
    }
  }

  val sh = new Function[util.List[NDList], NDList] {
    override def apply(t: util.List[NDList]): NDList = {
      val temp1 = t.get(1).get(0).mul(t.get(0).get(1)).sum(Array(-1))
      val temp2 = t.get(1).get(1).mul(t.get(0).get(1)).sum(Array(-1))
      val temp3 = t.get(1).get(2).mul(t.get(0).get(1)).sum(Array(-1))
      new NDList(NDArrays.stack(new NDList(temp1, temp2, temp3), -1))
    }
  }

  val f = new Function[util.List[NDList], NDList] {
    override def apply(t: util.List[NDList]): NDList = {
      val temp1 = t.get(1).get(0).mul(t.get(0).get(2)).sum(Array(-1))
      val temp2 = t.get(1).get(1).mul(t.get(0).get(2)).sum(Array(-1))
      val temp3 = t.get(1).get(2).mul(t.get(0).get(2)).sum(Array(-1))
      new NDList(NDArrays.stack(new NDList(temp1, temp2, temp3), -1))
    }
  }

  block2.add(new ParallelBlock(if (config.useDir && config.useSH) if (config.useTime && config.useFourier) shF else sh else if (config.useTime && config.useFourier) f else new Function[util.List[NDList], NDList] {
    override def apply(t: util.List[NDList]): NDList = new NDList(t.get(0).get(0), t.get(1).singletonOrThrow())
  }, List[Block](new LambdaBlock(new Function[NDList, NDList] {
    override def apply(t: NDList): NDList = t
  }), processBlock).asJava))

  //网络构造完毕
  //该模块输入为：点的位置、方向和时间
  //该模块输出为：密度和颜色

  def positionCode(input: NDArray, L: Int): NDArray = {
    //sin cos位置编码，L为编码阶数
    val output = new NDList(L * 2)
    for (i <- 0 until L) {
      val inputMulFactor = input.mul(Math.PI * (1 << i))
      output.add(inputMulFactor.sin())
      output.add(inputMulFactor.cos())
    }
    input.getNDArrayInternal.concat(output, -1)
  }

  def Fourier(t: NDArray, L: Int): NDArray = {
    //L阶傅里叶级数
    //t的范围是-1到1
    val outputList = new NDList(L * 2 + 1)
    outputList.add(t.onesLike())
    for (i <- 0 until L) {
      val tpi = t.mul((i + 1) * Math.PI)
      outputList.add(tpi.cos())
      outputList.add(tpi.sin())
    }
    NDArrays.concat(outputList, -1)
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
    //time：时间，最高维度尺寸为1，代表t，范围-1到1，没有则为null
    //training：如果在训练则拉高
    val output = block.forward(config.ps, new NDList(pos, time), training)
    val output2 = if (isCoarse && !training) null else block2.forward(config.ps, new NDList(output.get(1), dir, output.get(2)), training).singletonOrThrow()
    (output.get(0), output2)
    //返回：
    //density：密度，最高维度尺寸为1，代表密度，大于0
    //rgb：颜色，最高维度尺寸为3，代表r、g、b，范围0到1，如果非训练且为粗糙模型则输出null
  }

  def initialize(manager: NDManager): nerfModel = {
    block.initialize(manager, DataType.FLOAT32, new Shape(config.NSamples, config.batchNum, 3), new Shape(config.batchNum, 1))
    block2.initialize(manager, DataType.FLOAT32, new Shape(config.NSamples, config.batchNum, config.W), new Shape(config.batchNum, 3), new Shape(config.batchNum, 1 + 2 * config.fourierL))
    this
  }

  def save(os: DataOutputStream): Unit = {
    block.saveParameters(os)
    block2.saveParameters(os)
  }

  def load(manager: NDManager, is: DataInputStream): Unit = {
    block.loadParameters(manager, is)
    block2.loadParameters(manager, is)
  }
}