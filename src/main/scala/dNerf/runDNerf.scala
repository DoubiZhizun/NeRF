package dNerf

import ai.djl._
import ai.djl.engine.Engine
import ai.djl.modality.cv._
import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types._
import ai.djl.nn.Block
import ai.djl.training._
import ai.djl.training.dataset.Batch
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer._
import ai.djl.training.tracker._
import ai.djl.translate.Batchifier

import java.io._
import java.nio.file._
import scala.collection.mutable.ArrayBuffer

object runDNerf {

  def train(): Unit = {

    val config = dNerfConfig(
      device = Array(Device.gpu(0)),
      dataSetType = "blender",
      halfRes = true,
      testSkip = 8,
      useDir = true,
      useSH = true,
      useTime = true,
      useHierarchical = false,
      posL = 10,
      dirL = 4,
      fourierL = 10,
      D = 8,
      W = 256,
      skips = Array(4),
      NSamples = 64,
      NImportance = 64,
      rawNoiseStd = 1e0,
      whiteBkgd = true,
      linDisp = false,
      perturb = false,
      batchNum = 1024,
      lrate = 5e-4,
      lrateDecay = 250,
      addTvLoss = true,
      tvLossWeight = 1e-4,
      dataDir = "./data/dnerf_synthetic/lego",
      logDir = "./logs/lego_dSH",
      iPrint = 100,
      iImage = 500,
      iWeight = 50000,
      iTestSet = 50000,
      iVideo = 100000,
      NIter = 500000)

    val manager = NDManager.newBaseManager(config.device.head)

    val block = new dNerfBlock(config)
    val model = Model.newInstance("nerf")
    model.setBlock(block)
    val trainer = model.newTrainer(
      new DefaultTrainingConfig(Loss.l2Loss("L2Loss", 1))
        .optOptimizer(Optimizer.adam().optLearningRateTracker(Tracker.factor().setBaseValue(config.lrate.toFloat).setFactor(Math.pow(0.1, 1.0 / (config.lrateDecay * 1000)).toFloat).build()).optBeta1(0.9f).optBeta2(0.999f).optEpsilon(1e-7f).build())
        .addTrainingListeners(TrainingListener.Defaults.logging(): _*)
        .optDevices(config.device)
    )

    trainer.initialize(new Shape(config.batchNum, 3), new Shape(config.batchNum, 3), new Shape(config.batchNum, 2), new Shape(config.batchNum, 3), new Shape(config.batchNum, 1))

    val calculateLoss = if (config.useHierarchical) if (config.useTime && config.addTvLoss) (label: NDList, pred: NDList, loss: Loss) => loss.evaluate(label, new NDList(pred.get(0))).add(loss.evaluate(label, new NDList(pred.get(1)))).add(pred.get(2).pow(2).sum().mul(config.tvLossWeight)).add(pred.get(3).pow(2).sum().mul(config.tvLossWeight))
    else (label: NDList, pred: NDList, loss: Loss) => loss.evaluate(label, new NDList(pred.get(0))).add(loss.evaluate(label, new NDList(pred.get(1))))
    else if (config.useTime && config.addTvLoss) (label: NDList, pred: NDList, loss: Loss) => loss.evaluate(label, pred).add(pred.get(2).pow(2).sum().mul(config.tvLossWeight))
    else (label: NDList, pred: NDList, loss: Loss) => loss.evaluate(label, pred)

    val (trainDataSet, testDataSet, valDataSet, renderDataSet) = getDataSet(config, manager)

    val imageLogPaths = Paths.get(config.logDir, "imageLogs")
    val weightLogPaths = Paths.get(config.logDir, "weightLogs")
    val videoLogPaths = Paths.get(config.logDir, "videoLogs")
    Files.createDirectories(imageLogPaths)
    Files.createDirectories(weightLogPaths)
    Files.createDirectories(videoLogPaths)

    val logPs = new PrintStream(Paths.get(config.logDir, "log.txt").toString)
    System.setOut(logPs)

    print("Train start.\n")
    var idx = 0
    var stop = false
    var lossSum: Float = 0
    var valIterator = valDataSet.getData(manager).iterator()
    while (!stop) {
      val trainIterator = trainDataSet.getData(manager).iterator()
      while (trainIterator.hasNext && !stop) {
        val next = trainIterator.next()
        val splits = next.split(trainer.getDevices, false)
        val collector = manager.getEngine.newGradientCollector()
        for (s <- splits) {
          val inputs = s.getData
          val outputs = trainer.forward(new NDList(inputs.get(0), inputs.get(1), inputs.get(2), inputs.get(1), inputs.get(3), inputs.get(4)))
          val lossValue = calculateLoss(s.getLabels, outputs, trainer.getLoss)
          lossSum += lossValue.getFloat() / splits.length
          collector.backward(lossValue)
        }
        collector.close()
        trainer.step()
        next.close()
        idx += 1
        if (idx % config.iPrint == 0) {
          print(s"${idx} iterators train: loss is ${lossSum / config.iPrint}.\n")
          lossSum = 0
        }
        if (idx % config.iImage == 0) {
          print(s"${idx} iterators: log image.\n")
          if (!valIterator.hasNext) {
            valIterator = valDataSet.getData(manager).iterator()
          }
          val image = renderOneImage(valIterator, trainer, manager)
          val os = new FileOutputStream(Paths.get(imageLogPaths.toString, s"$idx.png").toString)
          image.save(os, "png")
          os.close()
          print("Log over.\n")
        }
        if (idx % config.iWeight == 0) {
          print(s"${idx} iterators: save weight.\n")
          val os = new DataOutputStream(new FileOutputStream(Paths.get(weightLogPaths.toString, s"$idx.npy").toString))
          block.saveParameters(os)
          os.close()
          print("Log over.\n")
        }
        if (idx % config.iTestSet == 0) {
          print(s"${idx} iterators: start to test.\n")
          var testIdx = 0
          var lossSumTest: Float = 0
          var lossSumTotal: Float = 0
          val testIterator = testDataSet.getData(manager).iterator()
          while (testIterator.hasNext) {
            val next = testIterator.next()
            val splits = next.split(trainer.getDevices, false)
            for (s <- splits) {
              val inputs = s.getData
              val outputs = trainer.forward(new NDList(inputs.get(0), inputs.get(1), inputs.get(2), inputs.get(1), inputs.get(3), inputs.get(4)))
              val lossValue = calculateLoss(s.getLabels, outputs, trainer.getLoss)
              lossSumTest += lossValue.getFloat() / splits.length
              lossSumTotal += lossValue.getFloat() / splits.length
            }
            next.close()
            testIdx += 1
            if (testIdx % config.iPrint == 0) {
              print(s"${testIdx} iterators test: loss is ${lossSumTest / config.iPrint}.\n")
              lossSumTest = 0
            }
          }
          print(s"Test over, mean loss is ${lossSumTotal / testIdx}.\n")
        }
        if (idx % config.iVideo == 0) {
          print(s"${idx} iterators: log video.\n")
          val images = renderImages(renderDataSet.getData(manager).iterator(), trainer, manager)
          val path = Paths.get(videoLogPaths.toString, s"$idx")
          Files.createDirectories(path)
          for (i <- images.indices) {
            val os = new FileOutputStream(Paths.get(path.toString, s"$i.png").toString)
            images(i).save(os, "png")
            os.close()
          }
          print("Log over.\n")
        }
        if (idx % config.NIter == 0) {
          print(s"${idx} iterators: train over.\n")
          stop = true
        }
      }
    }
    printf("Train over.\n")
    logPs.close()
    manager.close()
  }

  def renderImages(dataIterator: java.util.Iterator[Batch], trainer: Trainer, manager: NDManager): Array[Image] = {
    val buffer = new ArrayBuffer[Image]
    while (dataIterator.hasNext) {
      buffer.append(renderOneImage(dataIterator, trainer, manager))
    }
    buffer.toArray
  }

  def renderOneImage(dataIterator: java.util.Iterator[Batch], trainer: Trainer, manager: NDManager): Image = {
    var output: NDList = null
    var notDone = true
    val subManager = manager.newSubManager()
    do {
      val next = dataIterator.next()
      if (next.getProgress == 0) {
        output = new NDList(next.getProgressTotal.toInt)
      }
      val inputs = next.getData
      val evaluate = trainer.evaluate(new NDList(inputs.get(0), inputs.get(1), inputs.get(2), inputs.get(1), inputs.get(3), null)).get(0)
      evaluate.attach(subManager)
      output.add(evaluate)
      next.close()
      if (next.getProgress >= next.getProgressTotal - 1) {
        notDone = false
      }
    } while (notDone)

    val image = ImageFactory.getInstance().fromNDArray(NDArrays.stack(output, 0))
    subManager.close()
    image
  }

  def main(args: Array[String]): Unit = {
    train()
  }
}