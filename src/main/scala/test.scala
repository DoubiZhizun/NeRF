
import ai.djl.Device
import ai.djl.modality.cv.ImageFactory
import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.training.ParameterStore
import nerf._
import _root_.nerf.llff._

import java.nio.file.{Files, Paths}

object test {
  def main(args: Array[String]): Unit = {
    val manager = NDManager.newBaseManager(Device.cpu())
    //    val config = nerfConfig(
    //      device = Device.gpu(2),
    //      posL = 10,
    //      dirL = 4,
    //      useSH = true,
    //      useTime = false,
    //      timeL = 0,
    //      D = 8,
    //      W = 256,
    //      skips = Array(4),
    //      rawNoiseStd = 1e0,
    //      whiteBkgd = true,
    //      linDisp = false,
    //      NSamples = 64,
    //      NImportance = 64,
    //      perturb = false,
    //      batchNum = 1024,
    //      lrate = 5e-4,
    //      lrateDecay = 250,
    //      ndc = true,
    //      datadir = "./data/nerf_llff_data/fern",
    //      basedir = "./logs")
    val array = manager.arange(16).reshape(4, 4)
    val array2 = manager.arange(3).repeat(2).broadcast(4, 6)
    print(array.get(new NDIndex().addAllDim().addPickDim(array2)))
    //    print(array.matMul(array2))
    //    print(array2.sub(array))
    //    array.setRequiresGradient(true)
    //    val c = Engine.getInstance().newGradientCollector()
    //    val array2 = array.getNDArrayInternal.where(array.lt(0), manager.zeros(array.getShape, DataType.INT32))
    //    c.backward(array2)
    //    c.close()
    //    print(array.getGradient)
    //    val a = 1
    //    val (images, poses, bds, render_poses, i_test) = load_llff_data(basedir = ".\\data\\nerf_llff_data\\fern", factor = 8, recenter = true, bd_factor = .75, spherify = true, manager = manager)
    //    val i = images.toFloatArray
    //    val p = poses.toFloatArray
    //    val b = bds.toFloatArray
    //    val r = render_poses.toFloatArray
    //    print(i, p, b, r, i_test)

    //run_nerf.train(args)

    //print("1e-2".toDouble)
    //    val block = init_nerf_model(8, 256,  4, Array(4), true).asInstanceOf[SequentialBlock]
    //    block.initialize(manager, DataType.FLOAT32, new Shape(64, 120, 63), new Shape(64, 120, 27))
    //    print(block)
    //    val block = Linear.builder().setUnits(256).build()
    //    block.initialize(manager, DataType.FLOAT32, new Shape(80, 120, 160))
    //    print(block)
    //    val config = nerfConfig(
    //      coarseBlock = coreBlockGenerator.getBlock(),
    //      fineBlock = coreBlockGenerator.getBlock(),
    //      device = Device.cpu(),
    //      pos_L = 10,
    //      direction_L = 4,
    //      raw_noise_std = 1e0,
    //      white_bkgd = false,
    //      lindisp = false,
    //      N_samples = 64,
    //      N_importance = 64,
    //      perterb = false,
    //      N_rand = 1024,
    //      lrate = 5e-4,
    //      lrate_decay = 250,
    //      datadir = ".\\data\\nerf_llff_data\\fern",
    //      basedir = " .\\logs")
    //    val block = new nerf(config).getBlock()
    //    block.initialize(manager.newSubManager(), DataType.FLOAT32, new Shape(1, 3), new Shape(1, 3), new Shape(1, 2), new Shape(1, 3))
    //    val store = new ParameterStore()
    //    //    val collector = manager.getEngine.newGradientCollector()
    //    while (true) {
    //      val subManager = manager.newSubManager()
    //      val collector = subManager.getEngine.newGradientCollector()
    //      val c = block.forward(store, new NDList(subManager.create(new Shape(1024, 3)), subManager.create(new Shape(1024, 3)), subManager.create(new Shape(1024, 2)), subManager.create(new Shape(1024, 3))), true)
    //      collector.backward(c.get(0))
    //      collector.close()
    //      subManager.close()
    //    }
    //    val tst = manager.arange(1 << 20)
    //    tst.percentile(10)
    //    tst.close()
    //    val a = 1
    //    collector.backward(c.get(0))
    //    collector.close()
    //    print(c)
    //    val subManager = manager.newSubManager()
    //    val array = manager.arange(10)
    //    val array2 = subManager.arange(5)
    //    array.intern(array2)
    //    val a = 1
  }
}