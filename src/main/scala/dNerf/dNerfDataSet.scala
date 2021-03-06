package dNerf

import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.training.dataset._
import ai.djl.translate.Batchifier
import ai.djl.util.Progress
import dNerf.getDataSet._

import java._

final class dNerfDataSet(poses: NDArray, times: NDArray, hwf: Array[Float], images: NDArray, batchNum: Int) extends Dataset with java.lang.Iterable[Batch] with java.util.Iterator[Batch] {

  private var manager: NDManager = null
  private var dirs: NDArray = null

  private val poseNum = poses.getShape.get(0).toInt
  private val pixelSize = (hwf(0) * hwf(1)).toInt

  private var imagesHere: NDList = null

  require(pixelSize >= batchNum)

  private val idx = (0 until pixelSize).toArray

  override def getData(manager: NDManager): lang.Iterable[Batch] = {
    if (this.manager != null) {
      this.manager.close()
    }
    this.manager = manager.newSubManager()
    val dirs = getRaysDirs(hwf(0).toInt, hwf(1).toInt, hwf(2), this.manager)
    this.dirs = dirs.reshape(-1, 3)
    dirs.close()
    imagesHere = new NDList(poseNum)
    for (i <- 0 until poseNum) {
      val image = images.get(i)
      imagesHere.add(image.reshape(-1, 3))
      image.close()
    }
    imagesHere.attach(this.manager)
    this
  }

  override def prepare(progress: Progress): Unit = {}

  override def iterator(): util.Iterator[Batch] = this


  override def hasNext: Boolean = true

  override def next(): Batch = {
    val subManager = manager.newSubManager()

    for (i <- 0 until batchNum) {
      //打乱顺序
      val j = scala.util.Random.nextInt(pixelSize)
      val temp = idx(i)
      idx(i) = idx(j)
      idx(j) = temp
    }

    val indexArray = subManager.create(idx.slice(0, batchNum)).expandDims(-1).repeat(1, 3)

    val index = new NDIndex().addPickDim(indexArray)

    val poseIdx = scala.util.Random.nextInt(poseNum)

    val poseNow = poses.get(poseIdx)
    poseNow.attach(subManager)

    val batchDir = dirs.get(index)
    batchDir.attach(subManager)
    val batchRaysO = poseNow.get(":,-1")
    var batchRaysD = batchDir.matMul(poseNow.get(":,:3").transpose(1, 0))
    val batchRaysDNorm = batchRaysD.norm(Array(-1), true)
    batchRaysD = batchRaysD.div(batchRaysDNorm)
    val batchBounds = batchRaysDNorm.mul(2).concat(batchRaysDNorm.mul(6), -1)
    val batchTime = times.get(poseIdx)
    batchTime.attach(subManager)

    val batchImage = imagesHere.get(poseIdx).get(index)
    batchImage.attach(subManager)

    val data = new NDList(batchRaysO, batchRaysD, batchBounds, batchRaysD, batchTime)
    val label = new NDList(batchImage)

    new Batch(subManager, data, label, 1, Batchifier.STACK, Batchifier.STACK, 0, 1)
  }
}