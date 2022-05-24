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
      device = Array(Device.gpu(2)),
      dataSetType = "blender",
      halfRes = true,
      testSkip = 1,
      useDir = true,
      useSH = true,
      useTime = true,
      useFourier = false,
      useHierarchical = false,
      posL = 10,
      dirL = 4,
      timeL = 10,
      fourierL = 10,
      D = 8,
      W = 256,
      skips = Array(4),
      NSamples = 64,
      NImportance = 128,
      rawNoiseStd = 1e0,
      whiteBkgd = true,
      linDisp = false,
      perturb = false,
      batchNum = 1024,
      lrate = 5e-4,
      lrateDecay = 500,
      dataDir = "./data/dnerf_synthetic/mutant",
      logDir = "./logs/mutant",
      iPrint = 100,
      iImage = 500,
      iWeight = 50000,
      iTestSet = 50000,
      iVideo = 100000,
      trainIter = 800000,
      testIter = 1000)

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

    trainer.initialize(new Shape(config.batchNum, 3), new Shape(config.batchNum, 3), new Shape(config.batchNum, 2), new Shape(config.batchNum, 3), new Shape())

    val calculateLoss = if (config.useHierarchical) (label: NDList, pred: NDList, loss: Loss) => loss.evaluate(label, new NDList(pred.get(0))).add(loss.evaluate(label, new NDList(pred.get(1))))
    else (label: NDList, pred: NDList, loss: Loss) => loss.evaluate(label, new NDList(pred.get(0)))

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
    var lossSum: Float = 0
    val trainIterator = trainDataSet.getData(manager).iterator()
    val testIterator = testDataSet.getData(manager).iterator()
    valDataSet.getData(manager)
    renderDataSet.getData(manager)
    for (_ <- 0 until config.trainIter) {
      val next = trainIterator.next()
      val splits = next.split(trainer.getDevices, false)
      val collector = manager.getEngine.newGradientCollector()
      for (s <- splits) {
        val outputs = trainer.forward(s.getData)
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
        val image = renderOneImage(valDataSet, trainer, manager)
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
        var lossSumTotal: Float = 0
        for (_ <- 0 until config.testIter) {
          val next = testIterator.next()
          val splits = next.split(trainer.getDevices, false)
          for (s <- splits) {
            val outputs = trainer.forward(s.getData)
            val lossValue = calculateLoss(s.getLabels, outputs, trainer.getLoss)
            lossSumTotal += lossValue.getFloat() / splits.length
          }
          next.close()
        }
        print(s"Test over, mean loss is ${lossSumTotal / config.testIter}.\n")
      }
      if (idx % config.iVideo == 0) {
        print(s"${idx} iterators: log video.\n")
        val images = renderImages(renderDataSet, trainer, manager)
        val path = Paths.get(videoLogPaths.toString, s"$idx")
        Files.createDirectories(path)
        for (i <- images.indices) {
          val os = new FileOutputStream(Paths.get(path.toString, s"$i.png").toString)
          images(i).save(os, "png")
          os.close()
        }
        print("Log over.\n")
      }
      if (idx % config.trainIter == 0) {
        print(s"${idx} iterators: train over.\n")
      }
    }
    printf("Train over.\n")
    logPs.close()
    manager.close()
  }

  def renderImages(dataSet: dNerfRenderSet, trainer: Trainer, manager: NDManager): Array[Image] = {
    val buffer = new Array[Image](dataSet.getPoseNum)
    dataSet.setPoseIdx(0)
    for (i <- 0 until dataSet.getPoseNum) {
      buffer(i) = renderOneImage(dataSet, trainer, manager)
    }
    buffer
  }

  def renderOneImage(dataSet: dNerfRenderSet, trainer: Trainer, manager: NDManager): Image = {
    val output: NDList = new NDList(dataSet.getNumOfBatch)
    val subManager = manager.newSubManager()
    val iterator = dataSet.getData(manager).iterator()

    while (iterator.hasNext) {
      val next = iterator.next()
      val pred = trainer.evaluate(next.getData).get(0).mul(255).toType(DataType.UINT8, false)
      pred.attach(subManager)
      output.add(pred)
      next.close()
    }

    val image = ImageFactory.getInstance().fromNDArray(NDArrays.stack(output, 0))
    subManager.close()
    image
  }

  def main(args: Array[String]): Unit = {
    train()
  }
}