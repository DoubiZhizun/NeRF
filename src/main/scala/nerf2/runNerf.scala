package nerf2

import ai.djl._
import ai.djl.engine.Engine
import ai.djl.modality.cv._
import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types._
import ai.djl.nn.Block
import ai.djl.training._
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer._
import ai.djl.training.tracker._

import java.io._
import java.nio.file._

object runNerf {

  def train(): Unit = {

    val config = nerfConfig(
      device = Device.gpu(2),
      dataSetType = "llff",
      factor = 8,
      llffHold = 8,
      useDir = true,
      useTime = false,
      useHierarchical = true,
      postL = 10,
      dirL = 4,
      D = 8,
      W = 256,
      skips = Array(4),
      NSamples = 64,
      NImportance = 64,
      rawNoiseStd = 1e0,
      whiteBkgd = true,
      linDisp = false,
      perturb = false,
      ndc = true,
      batchNum = 1024,
      lrate = 5e-4,
      lrateDecay = 250,
      dataDir = "./data/nerf_llff_data/fern",
      logDir = "./logs",
      iPrint = 100,
      iImage = 500,
      iWeight = 10000,
      iTestSet = 50000,
      iVideo = 50000,
      NIter = 1000000)

    val manager = NDManager.newBaseManager(config.device)
    val adam = Optimizer.adam().optLearningRateTracker(Tracker.factor().setBaseValue(config.lrate.toFloat).setFactor(Math.pow(0.1, 1.0 / (config.lrateDecay * 1000)).toFloat).build()).optBeta1(0.9f).optBeta2(0.999f).optEpsilon(1e-7f).build()
    val ps = new ParameterStore(manager, false)
    ps.setParameterServer(manager.getEngine.newParameterServer(adam), Array(config.device))
    val loss = Loss.l2Loss("L2Loss", 1)

    val calculateLoss = if (config.useHierarchical) (label: NDList, pred: NDList) => loss.evaluate(label, new NDList(pred.get(0))).add(loss.evaluate(label, new NDList(pred.get(1))))
    else (label: NDList, pred: NDList) => loss.evaluate(label, pred)

    val block = new nerfBlock(config)

    val (trainDataSet, testDataSet, renderDataSet) = getDataSet(config, manager)

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
    while (!stop) {
      val trainIterator = trainDataSet.getData(manager).iterator()
      while (trainIterator.hasNext && !stop) {
        val next = trainIterator.next()
        val collector = manager.getEngine.newGradientCollector()
        val inputs = next.getData
        inputs.add(inputs.get(1))
        val outputs = block.forward(ps, inputs, true)
        val lossValue = calculateLoss(next.getLabels, outputs)
        lossSum += lossValue.getFloat()
        collector.backward(lossValue)
        collector.close()
        ps.updateAllParameters()
        next.close()
        idx += 1
        if (idx % config.iPrint == 0) {
          print(s"${idx} iterators train: loss is ${lossSum / config.iPrint}.\n")
          lossSum = 0
        }
        if (idx % config.iImage == 0) {
          val logWho = (idx / config.iImage - 1) % renderDataSet.get(0).getShape.get(0)
          print(s"${idx} iterators: log image is NO.${logWho}.\n")
          val index = new NDIndex().addSliceDim(logWho, logWho + 1)
          val logOne = new NDList(renderDataSet.get(0).get(index), renderDataSet.get(1).get(index), renderDataSet.get(2).get(index))
          val image = renderToImage(logOne, block, ps, manager)
          logOne.close()
          val os = new FileOutputStream(Paths.get(imageLogPaths.toString, s"$idx.png").toString)
          image(0).save(os, "png")
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
            val inputs = next.getData
            inputs.add(inputs.get(1))
            val outputs = block.forward(ps, inputs, false)
            val lossValue = loss.evaluate(next.getLabels, new NDList(outputs.get(0)))
            lossSumTest += lossValue.getFloat()
            lossSumTotal += lossValue.getFloat()
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
          val images = renderToImage(renderDataSet, block, ps, manager)
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

  def renderToImage(input: NDList, block: Block, ps: ParameterStore, manager: NDManager): Array[Image] = {
    //input内容为原点，方向和边界，都有四维，分别是图片数，图片宽，图片高和参数
    val output = new Array[Image](input.get(0).getShape.get(0).toInt)
    for (i <- 0 until input.get(0).getShape.get(0).toInt) {
      val imageManager = manager.newSubManager()
      val imageList = new NDList(input.get(0).getShape.get(1).toInt)
      for (j <- 0 until input.get(0).getShape.get(1).toInt) {
        val subManager = manager.newSubManager()
        val rays_o = input.get(0).get(i, j)
        val rays_d = input.get(1).get(i, j)
        val bounds = input.get(2).get(i, j)
        val netInput = new NDList(rays_o, rays_d, bounds, rays_d)
        netInput.attach(subManager)
        val outputImage = block.forward(ps, netInput, false).get(0).toType(DataType.UINT8, false)
        outputImage.attach(imageManager)
        imageList.add(outputImage)
        subManager.close()
      }
      val image = NDArrays.stack(imageList, 0)
      output(i) = ImageFactory.getInstance().fromNDArray(image)
      imageManager.close()
    }
    output
  }

  def main(args: Array[String]): Unit = {
    train()
  }
}