package nerf2

import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types._
import ai.djl.training.dataset._
import nerf2.llff._
import nerf2.blender._

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
    var poses, renderPoses, images, near, far: NDArray = null
    var hwf: Array[Float] = null
    var iTrain, iTest, iVal: Array[Int] = null
    if (config.dataSetType == "llff") {
      val subSubManager = subManager.newSubManager()
      val (poses2, renderPoses2, images2, bds) = loadLlffData(config.dataDir, config.factor, .75, subSubManager)
      new NDList(poses2, renderPoses2, images2, bds).attach(subManager)
      subSubManager.close()

      poses = poses2.get("...,:-1")
      renderPoses = renderPoses2
      images = images2

      hwf = poses2.get("0,:,-1").toFloatArray
      //hwf中的三项分别是高、宽、焦距

      near = bds.min().mul(.9)
      far = bds.max()

      iTest = (0 until(poses.getShape.get(0).toInt, config.llffHold)).toArray //测试集范围
      iVal = iTest //留档集即为测试集
      iTrain = (for (i <- 0 until poses.getShape.get(0).toInt if (!iTest.contains(i))) yield i).toArray //测试集以外的都是训练集
    } else if (config.dataSetType == "blender") {
      val subSubManager = subManager.newSubManager()
      val (poses2, renderPoses2, images2, hwf2, iSplit) = loadBlenderData(config.dataDir, config.halfRes, config.testSkip, subSubManager)
      new NDList(poses2, renderPoses2, images2).attach(subManager)
      subSubManager.close()

      poses = poses2
      renderPoses = renderPoses2
      images = images2
      images = images2
      hwf = hwf2
      near = subManager.create(2f)
      far = subManager.create(6f)

      iTrain = (0 until iSplit(0)).toArray
      iTest = (iSplit(0) until iSplit(1)).toArray
      iVal = (iSplit(1) until iSplit(2)).toArray
    } else if (config.dataSetType == "deepvoxels") {
      require(false, "还不支持deepvoxels数据集。\n")
      null
    } else {
      require(false, "数据集选择错误。\n")
      null
    }
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

    var raysOTrain = NDArrays.stack(raysOTrainList, 0).reshape(-1, 3)
    var raysDTrain = NDArrays.stack(raysDTrainList, 0).reshape(-1, 3)
    val labelTrain = NDArrays.stack(labelTrainList, 0).reshape(-1, 3)
    if (config.ndc) {
      val temp = ndc(hwf(0).toInt, hwf(1).toInt, hwf(2), 1, raysOTrain, raysDTrain)
      raysOTrain = temp._1
      raysDTrain = temp._2
    }
    val raysDTrainNorm = raysDTrain.norm(Array(-1), true)
    raysDTrain = raysDTrain.div(raysDTrainNorm)
    val boundsTrain = if (config.ndc) raysDTrainNorm.zerosLike().concat(raysDTrainNorm, -1) else raysDTrainNorm.mul(near).concat(raysDTrainNorm.mul(far), -1)

    var raysOTest = NDArrays.stack(raysOTestList, 0).reshape(-1, 3)
    var raysDTest = NDArrays.stack(raysDTestList, 0).reshape(-1, 3)
    val labelTest = NDArrays.stack(labelTestList, 0).reshape(-1, 3)
    if (config.ndc) {
      val temp = ndc(hwf(0).toInt, hwf(1).toInt, hwf(2), 1, raysOTest, raysDTest)
      raysOTest = temp._1
      raysDTest = temp._2
    }
    val raysDTestNorm = raysDTest.norm(Array(-1), true)
    raysDTest = raysDTest.div(raysDTestNorm)
    val boundsTest = if (config.ndc) raysDTestNorm.zerosLike().concat(raysDTestNorm, -1) else raysDTestNorm.mul(near).concat(raysDTestNorm.mul(far), -1)

    var raysOVal = NDArrays.stack(raysOValList, 0)
    var raysDVal = NDArrays.stack(raysDValList, 0)
    if (config.ndc) {
      val temp = ndc(hwf(0).toInt, hwf(1).toInt, hwf(2), 1, raysOVal, raysDVal)
      raysOVal = temp._1
      raysDVal = temp._2
    }
    val raysDValNorm = raysDVal.norm(Array(-1), true)
    raysDVal = raysDVal.div(raysDValNorm)
    val boundsVal = if (config.ndc) raysDValNorm.zerosLike().concat(raysDValNorm, -1) else raysDValNorm.mul(near).concat(raysDValNorm.mul(far), -1)

    var (renderRaysO, renderRaysD) = getRaysNp(hwf(0).toInt, hwf(1).toInt, hwf(2), renderPoses)
    if (config.ndc) {
      val temp = ndc(hwf(0).toInt, hwf(1).toInt, hwf(2), 1, renderRaysO, renderRaysD)
      renderRaysO = temp._1
      renderRaysD = temp._2
    }
    val renderRaysDNorm = renderRaysD.norm(Array(-1), true)
    renderRaysD = renderRaysD.div(renderRaysDNorm)
    val renderBounds = if (config.ndc) renderRaysDNorm.zerosLike().concat(renderRaysDNorm, -1) else renderRaysDNorm.mul(near).concat(renderRaysDNorm.mul(far), -1)

    raysOTrain.attach(manager)
    raysDTrain.attach(manager)
    boundsTrain.attach(manager)
    labelTrain.attach(manager)

    raysOTest.attach(manager)
    raysDTest.attach(manager)
    boundsTest.attach(manager)
    labelTest.attach(manager)

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