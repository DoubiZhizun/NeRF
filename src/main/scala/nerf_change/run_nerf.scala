package nerf_change

import ai.djl._
import ai.djl.metric._
import ai.djl.modality.cv._
import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.training._
import ai.djl.training.evaluator._
import ai.djl.training.listener._
import ai.djl.training.loss._
import ai.djl.training.optimizer._
import ai.djl.training.tracker._
import dataSetFnc._

import java.io._
import java.nio.file._

object run_nerf {

  def train(): Unit = {

    val config = nerfConfig(
      coarseBlock = coreBlockGenerator.getBlock(),
      fineBlock = coreBlockGenerator.getBlock(),
      device = Device.gpu(),
      pos_L = 10,
      direction_L = 4,
      raw_noise_std = 1e0,
      white_bkgd = false,
      lindisp = false,
      N_samples = 64,
      N_importance = 64,
      perterb = false,
      N_rand = 1024,
      lrate = 5e-4,
      lrate_decay = 250,
      ndc = true,
      datadir = "./data/nerf_llff_data/fern",
      basedir = "./logs")

    val manager = NDManager.newBaseManager(config.device)
    val (trainDataSet, testDataSet, trainDataSize, renderDataSet) = getDataSet(config, manager)

    val block = new nerf(config).getBlock()
    val model = Model.newInstance("nerf")
    model.setBlock(block)
    //model.load(Paths.get("./logs/nerf"), "nerf")
    val sgd = Optimizer.adam().optLearningRateTracker(Tracker.factor().setBaseValue(config.lrate.toFloat).setFactor(Math.pow(0.1, 1.0 / (config.lrate_decay * 1000)).toFloat).build()).optBeta1(0.9f).optBeta2(0.999f).optEpsilon(1e-7f).build()
    val trainer = model.newTrainer(new DefaultTrainingConfig(Loss.l2Loss("L2Loss", 1)).optOptimizer(sgd).addEvaluator(new Accuracy())
      .addTrainingListeners(TrainingListener.Defaults.logging(): _*).optDevices(Array(config.device)))
    trainer.initialize(new Shape(1, 3), new Shape(1, 3), new Shape(1, 2), new Shape(1, 3))
    trainer.setMetrics(new Metrics())

    for (i <- 0 until Math.ceil(200000f * config.N_rand / trainDataSize / 10).toInt) {
      printf(s"$i times train start.\n")
      EasyTrain.fit(trainer, 10, trainDataSet, testDataSet)
      val images = renderToImage(renderDataSet, trainer, manager)
      val paths = Paths.get(config.basedir, s"$i")
      Files.createDirectories(paths)
      for (j <- images.indices) {
        images(j).save(new FileOutputStream(Paths.get(paths.toString, s"$j.png").toString), "png")
      }
    }

    val modelDir = Paths.get("./logs/nerf")
    model.save(modelDir, "nerf")
  }

  def renderToImage(input: NDList, trainer: Trainer, manager: NDManager): Array[Image] = {
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
        val outputImage = trainer.evaluate(netInput).get(0).get("...,3:").stopGradient().mul(255).toType(DataType.UINT8, false)
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