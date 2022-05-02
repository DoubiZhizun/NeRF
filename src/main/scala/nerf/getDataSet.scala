package nerf

import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types._
import ai.djl.training.dataset._
import llff._

object getDataSet {

  def getRaysNp(H: Int, W: Int, focal: Float, c2w: NDArray): (NDArray, NDArray) = {
    val manager = c2w.getManager
    val i = manager.arange(0, W, 1, DataType.FLOAT32).reshape(1, W, 1).repeat(0, H)
    val j = manager.arange(0, H, 1, DataType.FLOAT32).reshape(H, 1, 1).repeat(1, W)
    val dirs = NDArrays.stack(new NDList(i.sub(W * .5).div(focal), j.sub(H * .5).div(focal).neg(), manager.ones(i.getShape).neg()), -1)
    val raysD = dirs.expandDims(-2).mul(c2w.get(":,:,:3")).sum(Array(-1))
    (c2w.get(":,:,-1").broadcast(raysD.getShape), raysD)
    //返回raysO和raysD
    //此时代表图片的维度为2
  }

  def apply(config: nerfConfig, manager: NDManager): (Dataset, Dataset, NDList) = {
    val subManager = manager.newSubManager()
    if (config.dataSetType == "llff") {
      var (poses, renderPoses, images, bds) = loadLlffData(config.dataDir, config.factor, .75, subManager)

      val hwf = poses.get("0,:3,-1").toFloatArray
      //hwf中的三项分别是高、宽、焦距

      poses = poses.get("...,:-1")
      //丢掉hwf的部分

      val near = bds.min().mul(.9)
      val far = bds.max()

      val (raysOFull, raysDFull) = getRaysNp(hwf(0).toInt, hwf(1).toInt, hwf(2), poses)

      val iTest = 0 until(poses.getShape.get(0).toInt, config.llffHold)

      val raysOTrainList = new NDList(poses.getShape.get(0).toInt - iTest.length)
      val raysDTrainList = new NDList(poses.getShape.get(0).toInt - iTest.length)
      val labelTrainList = new NDList(poses.getShape.get(0).toInt - iTest.length)
      val raysOTestList = new NDList(iTest.length)
      val raysDTestList = new NDList(iTest.length)
      val labelTestList = new NDList(iTest.length)

      for (i <- 0 until poses.getShape.get(0).toInt) {
        if (iTest.contains(i)) {
          raysOTestList.add(raysOFull.get(new NDIndex().addAllDim(2).addIndices(i)))
          raysDTestList.add(raysDFull.get(new NDIndex().addAllDim(2).addIndices(i)))
          labelTestList.add(images.get(i))
        } else {
          raysOTrainList.add(raysOFull.get(new NDIndex().addAllDim(2).addIndices(i)))
          raysDTrainList.add(raysDFull.get(new NDIndex().addAllDim(2).addIndices(i)))
          labelTrainList.add(images.get(i))
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

      var (renderRaysO, renderRaysD) = getRaysNp(hwf(0).toInt, hwf(1).toInt, hwf(2), renderPoses)
      renderRaysO = renderRaysO.transpose(2, 0, 1, 3)
      renderRaysD = renderRaysD.transpose(2, 0, 1, 3)
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
      val trainDataSet = new nerfDataSet(new NDList(raysOTrain, raysDTrain, boundsTrain), new NDList(labelTrain), config.batchNum)
      val testDataSet = new nerfDataSet(new NDList(raysOTest, raysDTest, boundsTest), new NDList(labelTest), config.batchNum)
      (trainDataSet, testDataSet, new NDList(renderRaysO, renderRaysD, renderBounds))
    } else if (config.dataSetType == "blender") {
      require(false, "还不支持blender数据集。\n")
      null
    } else if (config.dataSetType == "deepvoxels") {
      require(false, "还不支持deepvoxels数据集。\n")
      null
    } else {
      require(false, "数据集选择错误。\n")
      null
    }
  }
}