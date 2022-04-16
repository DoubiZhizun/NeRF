package nerf_change

import ai.djl._
import ai.djl.metric._
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types._
import ai.djl.training._
import ai.djl.training.evaluator._
import ai.djl.training.listener._
import ai.djl.training.loss._
import ai.djl.training.optimizer._
import ai.djl.training.tracker._
import dataSetFnc._

import java.nio.file.Paths

object run_nerf {

  def train(): Unit = {

    val config = nerfConfig(
      coarseBlock = coreBlockGenerator.getBlock(),
      fineBlock = coreBlockGenerator.getBlock(),
      device = Device.cpu(),
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
      datadir = ".\\data\\nerf_llff_data\\fern",
      basedir = " .\\logs")

    val manager = NDManager.newBaseManager(config.device)
    val (trainDataSet, testDataSet, trainDataSize) = getDataSet(config, manager)

    val block = new nerf(config).getBlock()
    val model = Model.newInstance("nerf")
    model.setBlock(block)
    val sgd = Optimizer.adam().optLearningRateTracker(Tracker.factor().setBaseValue(config.lrate.toFloat).setFactor(Math.pow(0.1, 1.0 / (config.lrate_decay * 1000)).toFloat).build()).optBeta1(0.9f).optBeta2(0.999f).optEpsilon(1e-7f).build()
    val trainer = model.newTrainer(new DefaultTrainingConfig(Loss.l2Loss("L2Loss", 1)).optOptimizer(sgd).optDevices(Array(config.device)).addTrainingListeners(TrainingListener.Defaults.logging(): _*))
    trainer.initialize(new Shape(1, 3), new Shape(1, 3), new Shape(1, 2), new Shape(1, 3))
    trainer.setMetrics(new Metrics())

    EasyTrain.fit(trainer, 1, trainDataSet, testDataSet)

    val modelDir = Paths.get(".\\logs\\nerf")
    model.save(modelDir, "nerf")
  }

  def main(args: Array[String]): Unit = {
    train()
  }
}