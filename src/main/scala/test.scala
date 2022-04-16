import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import nerf_origin.load_llff._

import scala.util.Random._
import com.sun.org.apache.xalan.internal.xsltc.cmdline.getopt.GetOpt
import nerf_origin._
import nerf_origin.run_nerf_helpers._
import ai.djl.nn.transformer.BertMaskedLanguageModelBlock._
import ai.djl.training.ParameterStore
import nerf_change._
import nerf_change.coreBlockGenerator._

import java.lang.reflect.Parameter

object test {
  def main(args: Array[String]): Unit = {
    val manager = NDManager.newBaseManager()
    //    val array = manager.arange(4).reshape(2, 2, 1)
    //    val array2 = manager.create(Array(1, 2)).reshape(2, 1)
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
    val tst = manager.arange(1 << 20)
    tst.percentile(10)
    tst.close()
    val a = 1
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