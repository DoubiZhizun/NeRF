package dNerf

import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types._
import ai.djl.training.dataset._
import dNerf.blender._

object getDataSet {
  def apply(config: dNerfConfig, manager: NDManager): (Dataset, Dataset, Dataset, Dataset) = {

    if (config.dataSetType == "blender") {
      val subManager = manager.newSubManager()
      val (poses, times, renderPoses, renderTimes, images, hwf, iSplit) = loadBlenderData(config.dataDir, config.halfRes, config.testSkip, subManager)

      val iTrainIndex = new NDIndex().addSliceDim(0, iSplit(0))
      val iTestIndex = new NDIndex().addSliceDim(iSplit(0), iSplit(1))
      val iValIndex = new NDIndex().addSliceDim(iSplit(1), iSplit(2))

      val trainPoses = poses.get(iTestIndex)
      val testPoses = poses.get(iTestIndex)
      val valPoses = poses.get(iValIndex)

      val trainTimes = times.get(iTrainIndex)
      val testTimes = times.get(iTestIndex)
      val valTimes = times.get(iValIndex)

      val trainImages = images.get(iTrainIndex)
      val testImages = images.get(iTestIndex)
      val valImages = images.get(iValIndex)

      new NDList(trainPoses, testPoses, valPoses, trainTimes, testTimes, valTimes, trainImages, testImages, valImages, renderPoses, renderTimes).attach(manager)
      subManager.close()
      (new dNerfDataSet(trainPoses, trainTimes, hwf, false, trainImages, config.batchNum),
        new dNerfDataSet(testPoses, testTimes, hwf, false, testImages, config.batchNum),
        new dNerfDataSet(valPoses, valTimes, hwf, true),
        new dNerfDataSet(renderPoses, renderTimes, hwf, true))
    } else {
      require(false, "数据集选择错误。\n")
      null
    }
  }
}