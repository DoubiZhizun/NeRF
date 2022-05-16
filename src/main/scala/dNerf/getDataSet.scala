package dNerf

import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types._
import ai.djl.training.dataset._
import dNerf.blender._

object getDataSet {
  def getRaysNp(H: Int, W: Int, focal: Float, c2w: NDArray): (NDArray, NDArray) = {
    //c2w：尺寸(3, 4)
    val manager = c2w.getManager
    val w = W * .5f / focal
    val h = H * .5f / focal
    val i = manager.linspace(-w, w, W).broadcast(H, W)
    val j = manager.linspace(-h, h, H).flip(0).reshape(H, 1).broadcast(H, W)
    val dirs = NDArrays.stack(new NDList(i, j, manager.full(i.getShape, -1, DataType.FLOAT32)), -1)
    val raysD = dirs.matMul(c2w.get(":,:3").transpose(1, 0))
    (c2w.get(":,-1"), raysD)
    //返回raysO和raysD
  }

  def getRaysDirs(H: Int, W: Int, focal: Float, manager: NDManager): NDArray = {
    val subManager = manager.newSubManager()
    val w = W * .5f / focal
    val h = H * .5f / focal
    val i = subManager.linspace(-w, w, W).broadcast(H, W)
    val j = subManager.linspace(-h, h, H).flip(0).reshape(H, 1).broadcast(H, W)
    val dirs = NDArrays.stack(new NDList(i, j, subManager.full(i.getShape, -1, DataType.FLOAT32)), -1)
    dirs.attach(manager)
    subManager.close()
    dirs
  }

  def apply(config: dNerfConfig, manager: NDManager): (dNerfDataSet, dNerfDataSet, dNerfRenderSet, dNerfRenderSet) = {

    if (config.dataSetType == "blender") {
      val subManager = manager.newSubManager()
      var (poses, times, renderPoses, renderTimes, images, hwf, iSplit) = loadBlenderData(config.dataDir, config.halfRes, config.testSkip, subManager)

      images = if (config.whiteBkgd) {
        val alpha = images.get("...,-1:")
        images.get("...,:3").mul(alpha).add(alpha.sub(1).neg())
      } else {
        images.get("...,:3")
      }

      val iTrainIndex = new NDIndex().addSliceDim(0, iSplit(0))
      val iTestIndex = new NDIndex().addSliceDim(iSplit(0), iSplit(1))
      val iValIndex = new NDIndex().addSliceDim(iSplit(1), iSplit(2))

      val trainPoses = poses.get(iTrainIndex)
      val testPoses = poses.get(iTestIndex)
      val valPoses = poses.get(iValIndex)

      val trainTimes = times.get(iTrainIndex)
      val testTimes = times.get(iTestIndex)
      val valTimes = times.get(iValIndex)

      val trainImages = images.get(iTrainIndex)
      val testImages = images.get(iTestIndex)

      new NDList(trainPoses, testPoses, valPoses, trainTimes, testTimes, valTimes, trainImages, testImages, renderPoses, renderTimes).attach(manager)
      subManager.close()
      (new dNerfDataSet(trainPoses, trainTimes, hwf, trainImages, config.batchNum, config.preCropFrac, config.preCropIter, config.preCropIterTime),
        new dNerfDataSet(testPoses, testTimes, hwf, testImages, config.batchNum),
        new dNerfRenderSet(valPoses, valTimes, hwf),
        new dNerfRenderSet(renderPoses, renderTimes, hwf))
    } else {
      require(false, "数据集选择错误。\n")
      null
    }
  }
}