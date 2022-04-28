package nerf_new3

import ai.djl._
import ai.djl.engine.Engine
import ai.djl.modality.cv._
import ai.djl.ndarray._
import ai.djl.ndarray.types._
import ai.djl.training._
import ai.djl.training.optimizer._
import ai.djl.training.tracker._
import nerf_new3.dataSetFnc._

import java.io._
import java.nio.file._

object run_nerf {

  def train(): Unit = {

    val config = nerfConfig(
      device = Device.gpu(1),
      pos_L = 10,
      raw_noise_std = 1e0,
      lindisp = false,
      N_samples = 64,
      perterb = false,
      lrate = 5e-4,
      lrate_decay = 250,
      ndc = true,
      Mf = 128,
      factor = 4,
      datadir = "./data/nerf_llff_data/fern",
      basedir = "./logs")

    val manager = NDManager.newBaseManager(config.device)
    val adam = Optimizer.adam().optLearningRateTracker(Tracker.factor().setBaseValue(config.lrate.toFloat).setFactor(Math.pow(0.1, 1.0 / (config.lrate_decay * 1000)).toFloat).build()).optBeta1(0.9f).optBeta2(0.999f).optEpsilon(1e-7f).build()
    config.ps = new ParameterStore(manager, false)
    config.ps.setParameterServer(Engine.getInstance().newParameterServer(adam), Array(config.device))
    config.block = new mcBlock(config, manager)

    val model = new nerf(config)

    val (dataSet, hwf, renderDataSet) = getDataSet(config, manager)

    print("Start to train.\n")
    var idx = 0
    for (_ <- 0 until 500) {
      val iterator = dataSet.getData(manager).iterator()
      while (iterator.hasNext) {
        val next = iterator.next()
        val loss = model.train(hwf(0).toInt, hwf(1).toInt, hwf(2), next.getData.get(0), next.getData.get(1), images = next.getLabels.get(0))
        next.close()
        idx += 1
        if(idx % 50 == 0){
          print(s"${idx} iterators train: loss is ${loss}.\n")
        }
        if (idx % 2500 == 0) {
          print("Start to render.\n")
          model.noise(false)
          val images = renderToImage(renderDataSet, hwf, model, manager)
          val paths = Paths.get(config.basedir, s"${idx / 2500}")
          Files.createDirectories(paths)
          for (j <- images.indices) {
            images(j).save(new FileOutputStream(Paths.get(paths.toString, s"$j.png").toString), "png")
          }
          model.noise(true)
          print("Render over.\n")
        }
      }
    }
  }

  def renderToImage(input: NDList, hwf: Array[Float], model: nerf, manager: NDManager): Array[Image] = {
    //input内容为原点，方向和边界，都有四维，分别是图片数，图片宽，图片高和参数
    val output = new Array[Image](input.get(0).getShape.get(0).toInt)
    for (i <- 0 until input.get(0).getShape.get(0).toInt) {
      val imageManager = manager.newSubManager()
      val render = new NDList(input.get(0).get(i), input.get(1).get(i))
      render.attach(imageManager)
      val image = model.predict(hwf(0).toInt, hwf(1).toInt, hwf(2), render.get(0), render.get(1)).mul(255).toType(DataType.UINT8, false)
      output(i) = ImageFactory.getInstance().fromNDArray(image)
      imageManager.close()
    }
    output
  }

  def main(args: Array[String]): Unit = {
    train()
  }
}