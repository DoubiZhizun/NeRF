package hashGridNerf

import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types._
import ai.djl.training.dataset._
import hashGridNerf.blender._

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

  def apply(config: nerfConfig, manager: NDManager): (Dataset, Dataset, NDList, NDList) = {
    val subManager = manager.newSubManager()
    var (poses, renderPoses, images, hwf, iSplit) = loadBlenderData(config.dataDir, config.halfRes, config.testSkip, subManager)

    poses.set(new NDIndex(":,:,-1"), poses.get(":,:,-1").sub(2).div(4))
    renderPoses.set(new NDIndex(":,:,-1"), renderPoses.get(":,:,-1").sub(2).div(4))
    //将-2到2的空间变换到0到1，只需要将观察点所有坐标+2然后再除4

    images = if (config.whiteBkgd) {
      val alpha = images.get("...,-1:")
      images.get("...,:3").mul(alpha).add(alpha.sub(1).neg())
    } else {
      images.get("...,:3")
    }

    val near = subManager.create(0.5f)
    val far = subManager.create(1.5f)

    val iTrain = (0 until iSplit(0)).toArray
    val iVal = (iSplit(0) until iSplit(1)).toArray
    val iTest = (iSplit(1) until iSplit(2)).toArray

    val (raysOFull, raysDFull) = getRaysNp(hwf(0).toInt, hwf(1).toInt, hwf(2), poses)

    val raysOTrainList = new NDList(iTrain.length)
    val raysDTrainList = new NDList(iTrain.length)
    val labelTrainList = new NDList(iTrain.length)
    val raysOTestList = new NDList(iTest.length)
    val raysDTestList = new NDList(iTest.length)
    val labelTestList = new NDList(iTest.length)
    val raysOValList = new NDList(iVal.length)
    val raysDValList = new NDList(iVal.length)

    for (i <- 0 until poses.getShape.get(0).toInt) {
      val raysONow = raysOFull.get(i)
      val raysDNow = raysDFull.get(i)
      val labelNow = images.get(i)
      if (iTrain.contains(i)) {
        raysOTrainList.add(raysONow)
        raysDTrainList.add(raysDNow)
        labelTrainList.add(labelNow)
      }
      if (iTest.contains(i)) {
        raysOTestList.add(raysONow)
        raysDTestList.add(raysDNow)
        labelTestList.add(labelNow)
      }
      if (iVal.contains(i)) {
        raysOValList.add(raysOFull.get(i))
        raysDValList.add(raysDFull.get(i))
      }
    }

    val raysOTrain = NDArrays.stack(raysOTrainList, 0).reshape(-1, 3)
    var raysDTrain = NDArrays.stack(raysDTrainList, 0).reshape(-1, 3)
    val labelTrain = NDArrays.stack(labelTrainList, 0).reshape(-1, 3)
    val raysDTrainNorm = raysDTrain.norm(Array(-1), true)
    raysDTrain = raysDTrain.div(raysDTrainNorm)
    val boundsTrain = raysDTrainNorm.mul(near).concat(raysDTrainNorm.mul(far), -1)

    val raysOTest = NDArrays.stack(raysOTestList, 0).reshape(-1, 3)
    var raysDTest = NDArrays.stack(raysDTestList, 0).reshape(-1, 3)
    val labelTest = NDArrays.stack(labelTestList, 0).reshape(-1, 3)
    val raysDTestNorm = raysDTest.norm(Array(-1), true)
    raysDTest = raysDTest.div(raysDTestNorm)
    val boundsTest = raysDTestNorm.mul(near).concat(raysDTestNorm.mul(far), -1)

    val raysOVal = NDArrays.stack(raysOValList, 0)
    var raysDVal = NDArrays.stack(raysDValList, 0)
    val raysDValNorm = raysDVal.norm(Array(-1), true)
    raysDVal = raysDVal.div(raysDValNorm)
    val boundsVal = raysDValNorm.mul(near).concat(raysDValNorm.mul(far), -1)

    var (renderRaysO, renderRaysD) = getRaysNp(hwf(0).toInt, hwf(1).toInt, hwf(2), renderPoses)
    val renderRaysDNorm = renderRaysD.norm(Array(-1), true)
    renderRaysD = renderRaysD.div(renderRaysDNorm)
    val renderBounds = renderRaysDNorm.mul(near).concat(renderRaysDNorm.mul(far), -1)

    raysOTrain.attach(manager)
    raysDTrain.attach(manager)
    boundsTrain.attach(manager)
    labelTrain.attach(manager)

    raysOTest.attach(manager)
    raysDTest.attach(manager)
    boundsTest.attach(manager)
    labelTest.attach(manager)

    raysOVal.attach(manager)
    raysDVal.attach(manager)
    boundsVal.attach(manager)

    renderRaysO.attach(manager)
    renderRaysD.attach(manager)
    renderBounds.attach(manager)

    subManager.close()
    val trainDataSet = //new ArrayDataset.Builder().setData(raysOTrain, raysDTrain, boundsTrain).optLabels(labelTrain).setSampling(config.batchNum, true).build()
      new nerfDataSet(new NDList(raysOTrain, raysDTrain, boundsTrain), new NDList(labelTrain), config.batchNum)
    val testDataSet = //new ArrayDataset.Builder().setData(raysOTest, raysDTest, boundsTest).optLabels(labelTest).setSampling(config.batchNum, true).build()
      new nerfDataSet(new NDList(raysOTest, raysDTest, boundsTest), new NDList(labelTest), config.batchNum)
    (trainDataSet, testDataSet, new NDList(raysOVal, raysDVal, boundsVal), new NDList(renderRaysO, renderRaysD, renderBounds))
  }
}