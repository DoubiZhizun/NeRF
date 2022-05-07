package nerf

import ai.djl._
import ai.djl.engine.Engine
import ai.djl.metric._
import ai.djl.modality.cv._
import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types._
import ai.djl.training._
import ai.djl.training.evaluator._
import ai.djl.training.listener._
import ai.djl.training.loss._
import ai.djl.training.optimizer._
import ai.djl.training.tracker._
import nerf.getDataSet._

import java.io._
import java.nio.file._

object runNerf {

  def train(config: nerfConfig): Unit = {

    val manager = NDManager.newBaseManager(config.device)
    val adam = Optimizer.adam().optLearningRateTracker(Tracker.factor().setBaseValue(config.lrate.toFloat).setFactor(Math.pow(0.1, 1.0 / (config.lrateDecay * 1000)).toFloat).build()).optBeta1(0.9f).optBeta2(0.999f).optEpsilon(1e-7f).build()
    config.ps = new ParameterStore(manager, false)
    config.ps.setParameterServer(Engine.getInstance().newParameterServer(adam), Array(config.device))

    val model = new nerf(config, manager)

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
        lossSum += model.train(next.getData.get(0), next.getData.get(1), null, next.getData.get(2), next.getData.get(1), next.getLabels.get(0))
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
          val image = renderToImage(logOne, model, manager)
          logOne.close()
          val os = new FileOutputStream(Paths.get(imageLogPaths.toString, s"$idx.png").toString)
          image(0).save(os, "png")
          os.close()
          print("Log over.\n")
        }
        if (idx % config.iWeight == 0) {
          print(s"${idx} iterators: save weight.\n")
          val os = new DataOutputStream(new FileOutputStream(Paths.get(weightLogPaths.toString, s"$idx.npy").toString))
          model.save(os)
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
            val output = model.predict(next.getData.get(0), next.getData.get(1), null, next.getData.get(2), next.getData.get(1))
            val loss = model.loss.evaluate(new NDList(output), new NDList(next.getLabels.get(0))).getFloat()
            next.close()
            lossSumTest += loss
            lossSumTotal += loss
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
          val images = renderToImage(renderDataSet, model, manager)
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

  def renderToImage(input: NDList, model: nerf, manager: NDManager): Array[Image] = {
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
        val outputImage = model.predict(netInput.get(0), netInput.get(1), null, netInput.get(2), netInput.get(3)).mul(255).toType(DataType.UINT8, false)
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
    val config1 = nerfConfig(
      device = Device.gpu(1),
      dataSetType = "llff",
      factor = 8,
      llffHold = 8,
      useDir = true,
      useSH = true,
      useTime = false,
      useFourier = false,
      fourierL = 5,
      useHierarchical = true,
      posL = 20,
      timeL = 10,
      dirL = 4,
      D = 8,
      W = 256,
      skips = Array(4),
      NSamples = 64,
      NImportance = 64,
      rawNoiseStd = 1e0,
      whiteBkgd = false,
      linDisp = false,
      perturb = false,
      ndc = true,
      batchNum = 1024,
      lrate = 5e-4,
      lrateDecay = 500,
      dataDir = "./data/nerf_llff_data/fern",
      logDir = "./SHlogs",
      iPrint = 100,
      iImage = 500,
      iWeight = 10000,
      iTestSet = 50000,
      iVideo = 50000,
      NIter = 500000)

    val config2 = nerfConfig(
      device = Device.gpu(1),
      dataSetType = "llff",
      factor = 8,
      llffHold = 8,
      useDir = true,
      useSH = false,
      useTime = false,
      useFourier = false,
      fourierL = 5,
      useHierarchical = true,
      posL = 10,
      timeL = 10,
      dirL = 4,
      D = 8,
      W = 256,
      skips = Array(4),
      NSamples = 64,
      NImportance = 64,
      rawNoiseStd = 1e0,
      whiteBkgd = false,
      linDisp = false,
      perturb = false,
      ndc = true,
      batchNum = 1024,
      lrate = 5e-4,
      lrateDecay = 250,
      dataDir = "./data/nerf_llff_data/fern",
      logDir = "./NSHlogs",
      iPrint = 100,
      iImage = 500,
      iWeight = 10000,
      iTestSet = 50000,
      iVideo = 50000,
      NIter = 500000)

    //train(config2)
    train(config1)
  }
}