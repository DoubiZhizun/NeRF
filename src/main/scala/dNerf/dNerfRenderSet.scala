package dNerf

import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.training.dataset._
import ai.djl.translate.Batchifier
import ai.djl.util.Progress
import dNerf.getDataSet._

import java._

final class dNerfRenderSet(poses: NDArray, times: NDArray, hwf: Array[Float]) extends Dataset with java.lang.Iterable[Batch] with java.util.Iterator[Batch] {

  private var manager: NDManager = null
  private var dirs: NDArray = null

  private val poseNum = poses.getShape.get(0).toInt
  private var poseIdx = 0
  private var poseNow: NDArray = null

  private val numOfBatch = hwf(0).toInt
  private var idx = 0

  def getPoseNum: Int = poseNum

  def setPoseIdx(idx: Int): dNerfRenderSet = {
    poseIdx = idx
    this
  }

  def getPoseIdx: Int = poseIdx

  def getNumOfBatch: Int = numOfBatch

  override def getData(manager: NDManager): lang.Iterable[Batch] = {
    if (this.manager != null) {
      this.manager.close()
    }
    this.manager = manager.newSubManager()
    dirs = getRaysDirs(hwf(0).toInt, hwf(1).toInt, hwf(2), this.manager)
    this
  }

  override def prepare(progress: Progress): Unit = {}

  override def iterator(): util.Iterator[Batch] = {
    poseIdx += 1
    if (poseIdx >= poseNum) {
      poseIdx = 0
    }
    idx = 0
    this
  }

  override def hasNext: Boolean = idx < numOfBatch

  override def next(): Batch = {
    val subManager = manager.newSubManager()

    val poseNow = poses.get(poseIdx)
    poseNow.attach(subManager)

    val batchDir = dirs.get(idx)
    batchDir.attach(subManager)
    val batchRaysO = poseNow.get(":,-1")
    var batchRaysD = batchDir.matMul(poseNow.get(":,:3").transpose(1, 0))
    val batchRaysDNorm = batchRaysD.norm(Array(-1), true)
    batchRaysD = batchRaysD.div(batchRaysDNorm)
    val batchBounds = batchRaysDNorm.mul(2).concat(batchRaysDNorm.mul(6), -1)
    val batchTime = times.get(poseIdx)
    batchTime.attach(subManager)

    val data = new NDList(batchRaysO, batchRaysD, batchBounds, batchRaysD, batchTime)
    val label = new NDList(0)

    idx += 1
    new Batch(subManager, data, label, 1, Batchifier.STACK, Batchifier.STACK, idx, numOfBatch)
  }
}