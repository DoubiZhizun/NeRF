package dNerf2

import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types._
import ai.djl.training.dataset._
import dNerf.blender._

object getDataSet {

  def getRaysNp(H: Int, W: Int, focal: Float, c2w: NDArray): (NDArray, NDArray) = {
    val manager = c2w.getManager
    val i = manager.arange(0, W, 1, DataType.FLOAT32).reshape(1, 1, W).repeat(1, H)
    val j = manager.arange(0, H, 1, DataType.FLOAT32).reshape(1, H, 1).repeat(2, W)
    val dirs = NDArrays.stack(new NDList(i.sub(W * .5).div(focal), j.sub(H * .5).div(focal).neg(), manager.full(i.getShape, -1, DataType.FLOAT32)), -1)
    val raysD = dirs.matMul(c2w.get(":,:,:3").transpose(0, 2, 1).expandDims(1))
    (c2w.get(":,:,-1").expandDims(1).expandDims(1).broadcast(raysD.getShape), raysD)
    //返回raysO和raysD
    //此时代表图片的维度为2
  }

  def apply(config: dNerfConfig, manager: NDManager): (Dataset, Dataset, NDList, NDList) = {
    val subManager = manager.newSubManager()
    var poses, times, renderPoses, renderTimes, images, near, far: NDArray = null
    var hwf: Array[Float] = null
    var iSplit: Array[NDIndex] = null
    if (config.dataSetType == "blender") {
      val subSubManager = subManager.newSubManager()
      val (poses2, times2, renderPoses2, renderTimes2, images2, hwf2, iSplit2) = loadBlenderData(config.dataDir, config.halfRes, config.testSkip, subSubManager)
      new NDList(poses2, times2, renderPoses2, renderTimes2, images2).attach(subManager)
      subSubManager.close()

      poses = poses2
      times = times2
      renderPoses = renderPoses2
      renderTimes = renderTimes2
      images = if (config.whiteBkgd) {
        val alpha = images2.get("...,-1:")
        images2.get("...,:3").mul(alpha).add(alpha.sub(1).neg())
      } else {
        images2.get("...,:3")
      }
      hwf = hwf2
      near = subManager.create(2f)
      far = subManager.create(6f)

      iSplit = new Array[NDIndex](3)
      iSplit(0) = new NDIndex().addSliceDim(0, iSplit2(0))
      iSplit(1) = new NDIndex().addSliceDim(iSplit2(0), iSplit2(1))
      iSplit(2) = new NDIndex().addSliceDim(iSplit2(1), iSplit2(2))
    } else {
      require(false, "数据集选择错误。\n")
    }
    val (raysOFull, raysDFull) = getRaysNp(hwf(0).toInt, hwf(1).toInt, hwf(2), poses)

    val raysOTrain = raysOFull.get(iSplit(0)).reshape(-1, 3)
    var raysDTrain = raysDFull.get(iSplit(0)).reshape(-1, 3)
    val labelTrain = images.get(iSplit(0)).reshape(-1, 3)
    val timesTrain = times.get(iSplit(0)).reshape(-1, 1).repeat(0, (hwf(0) * hwf(1)).toInt)
    val raysDTrainNorm = raysDTrain.norm(Array(-1), true)
    raysDTrain = raysDTrain.div(raysDTrainNorm)
    val boundsTrain = raysDTrainNorm.mul(near).concat(raysDTrainNorm.mul(far), -1)

    val raysOTest = raysOFull.get(iSplit(1)).reshape(-1, 3)
    var raysDTest = raysDFull.get(iSplit(1)).reshape(-1, 3)
    val labelTest = images.get(iSplit(1)).reshape(-1, 3)
    val timesTest = times.get(iSplit(1)).reshape(-1, 1).repeat(0, (hwf(0) * hwf(1)).toInt)
    val raysDTestNorm = raysDTest.norm(Array(-1), true)
    raysDTest = raysDTest.div(raysDTestNorm)
    val boundsTest = raysDTestNorm.mul(near).concat(raysDTestNorm.mul(far), -1)

    val raysOVal = raysOFull.get(iSplit(2))
    var raysDVal = raysDFull.get(iSplit(2))
    val timesVal = times.get(iSplit(2)).reshape(-1, 1)
    val raysDValNorm = raysDVal.norm(Array(-1), true)
    raysDVal = raysDVal.div(raysDValNorm)
    val boundsVal = raysDValNorm.mul(near).concat(raysDValNorm.mul(far), -1)

    var (renderRaysO, renderRaysD) = getRaysNp(hwf(0).toInt, hwf(1).toInt, hwf(2), renderPoses)
    val renderRaysDNorm = renderRaysD.norm(Array(-1), true)
    renderRaysD = renderRaysD.div(renderRaysDNorm)
    val renderBounds = renderRaysDNorm.mul(near).concat(renderRaysDNorm.mul(far), -1)
    renderTimes = renderTimes.reshape(-1, 1)

    raysOTrain.attach(manager)
    raysDTrain.attach(manager)
    timesTrain.attach(manager)
    boundsTrain.attach(manager)
    labelTrain.attach(manager)

    raysOTest.attach(manager)
    raysDTest.attach(manager)
    timesTest.attach(manager)
    boundsTest.attach(manager)
    labelTest.attach(manager)

    raysOVal.attach(manager)
    raysDVal.attach(manager)
    timesVal.attach(manager)
    boundsVal.attach(manager)

    renderRaysO.attach(manager)
    renderRaysD.attach(manager)
    renderTimes.attach(manager)
    renderBounds.attach(manager)

    subManager.close()
    val trainDataSet = //new ArrayDataset.Builder().setData(raysOTrain, raysDTrain, boundsTrain).optLabels(labelTrain).setSampling(config.batchNum, true).build()
      new nerfDataSet(new NDList(raysOTrain, raysDTrain, boundsTrain, timesTrain), new NDList(labelTrain), config.batchNum)
    val testDataSet = //new ArrayDataset.Builder().setData(raysOTest, raysDTest, boundsTest).optLabels(labelTest).setSampling(config.batchNum, true).build()
      new nerfDataSet(new NDList(raysOTest, raysDTest, boundsTest, timesTest), new NDList(labelTest), config.batchNum)
    (trainDataSet, testDataSet, new NDList(raysOVal, raysDVal, boundsVal, timesVal), new NDList(renderRaysO, renderRaysD, renderBounds, renderTimes))
  }
}