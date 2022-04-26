package nerf_new

import ai.djl._
import ai.djl.engine.Engine
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
import nerf_new.dataSetFnc._

import java.io._
import java.nio.file._

object run_nerf {

  def train(): Unit = {

    val config = nerfConfig(
      device = Device.gpu(1),
      pos_L = 10,
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
    val adam = Optimizer.adam().optLearningRateTracker(Tracker.factor().setBaseValue(config.lrate.toFloat).setFactor(Math.pow(0.1, 1.0 / (config.lrate_decay * 1000)).toFloat).build()).optBeta1(0.9f).optBeta2(0.999f).optEpsilon(1e-7f).build()
    val ps = new ParameterStore(manager, false)
    ps.setParameterServer(Engine.getInstance().newParameterServer(adam), Array(config.device))
    config.coarseBlock = new cfBlock(config, manager, ps)
    config.fineBlock = new cfBlock(config, manager, ps)

    val model = new nerf(config, ps)

    val (dataSet, trainDataSize, renderDataSet) = getDataSet(config, manager)

    print("Start to train.\n")
    var idx = 0
    for (_ <- 0 until Math.ceil(200000f * config.N_rand / trainDataSize).toInt) {
      val iterator = dataSet.getData(manager).iterator()
      while (iterator.hasNext) {
        val next = iterator.next()
        val loss = model.train(next.getData.get(0), next.getData.get(1), next.getData.get(2), next.getData.get(3), next.getData.get(4), next.getLabels.get(0))
        next.close()
        idx += 1
        if (idx % 1000 == 0) {
          print(s"${idx} iterators train: loss is ${loss}.\n")
        }
        if (idx % 50000 == 0) {
          print("Start to render.\n")
          model.perturb(false)
          val images = renderToImage(renderDataSet, model, manager)
          val paths = Paths.get(config.basedir, s"${idx / 50000}")
          Files.createDirectories(paths)
          for (j <- images.indices) {
            images(j).save(new FileOutputStream(Paths.get(paths.toString, s"$j.png").toString), "png")
          }
          model.perturb(true)
          print("Render over.\n")
        }
      }
    }
    print("Train over.\n")
    print("Start to final render.\n")
    model.perturb(false)
    val images = renderToImage(renderDataSet, model, manager)
    val paths = Paths.get(config.basedir, "final")
    Files.createDirectories(paths)
    for (j <- images.indices) {
      images(j).save(new FileOutputStream(Paths.get(paths.toString, s"$j.png").toString), "png")
    }
    model.perturb(true)
    print("Render over.\n")
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
        val near = input.get(2).get(i, j)
        val far = input.get(3).get(i, j)
        val viewdirs = input.get(4).get(i, j)
        val netInput = new NDList(rays_o, rays_d, near, far, viewdirs)
        netInput.attach(subManager)
        val outputImage = model.predict(netInput.get(0), netInput.get(1), netInput.get(2), netInput.get(3), netInput.get(4)).mul(255).toType(DataType.UINT8, false)
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