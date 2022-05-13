package dNerf

import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.ndarray.types.DataType
import ai.djl.training.dataset._
import ai.djl.translate.Batchifier
import ai.djl.util.Progress
import dNerf.dNerfDataSet.getRaysNp

import java._

final class dNerfDataSet(poses: NDArray, times: NDArray, hwf: Array[Float], isRender: Boolean, images: NDArray = null, batchNum: Int = 0) extends Dataset with java.lang.Iterable[Batch] with java.util.Iterator[Batch] {
  //isRender位true是代表该数据集为渲染集
  //渲染集不提供image、每个batch提供图像的一列，且不会随机打乱

  var manager: NDManager = null

  val poseNum = poses.getShape.get(0).toInt
  var poseIdx = 0
  var poseIdxNear = if (isRender) null else new Array[Int](2)

  val pixelSize = (hwf(0) * hwf(1)).toInt
  var pixelIdx = 0

  val pixelStep = if (isRender) hwf(1).toInt else batchNum

  val numOfBatch = if (isRender) hwf(0).toInt else Math.ceil(pixelSize.toDouble / batchNum).toInt
  var batchIdx = 0

  var managerNow: NDManager = null
  var dataNow: NDList = null
  var imageNow: NDArray = null

  private def newPixels(): Unit = {
    managerNow = manager.newSubManager()

    val managerTemp = manager.newSubManager()
    val posesNow = poses.get(poseIdx)
    val timesNow = times.get(poseIdx)
    new NDList(posesNow, timesNow).attach(managerTemp)
    var (raysONow, raysDNow) = getRaysNp(hwf(0).toInt, hwf(1).toInt, hwf(2), posesNow)
    val raysDNowNorm = raysDNow.norm(Array(-1), true)
    raysDNow = raysDNow.div(raysDNowNorm)
    var boundsNow = raysDNowNorm.mul(.2).concat(raysDNowNorm.mul(.6), -1)
    //得到raysD、bounds和times

    if (!isRender) {
      //如果不是渲染，则需要打乱
      imageNow = images.get(poseIdx)
      new NDList(imageNow).attach(managerTemp)
      imageNow = imageNow.reshape(-1, 3)

      val list = (0 until pixelSize).toArray
      for (i <- 0 until pixelSize) {
        val j = scala.util.Random.nextInt(pixelSize)
        val temp = list(j)
        list(j) = list(i)
        list(i) = temp
      }
      val ndArrayList = managerTemp.create(list).expandDims(-1)

      //打乱
      raysDNow = raysDNow.get(new NDIndex().addPickDim(ndArrayList.broadcast(raysDNow.getShape)))
      boundsNow = boundsNow.get(new NDIndex().addPickDim(ndArrayList.broadcast(boundsNow.getShape)))
      imageNow = imageNow.get(new NDIndex().addPickDim(ndArrayList.broadcast(imageNow.getShape)))

      new NDList(imageNow).attach(managerNow)

      poseIdxNear(0) = if (poseIdx > 0) poseIdx - 1 else 1
      poseIdxNear(1) = if (poseIdx < poseNum - 1) poseIdx + 1 else poseNum - 2
    }
    dataNow = new NDList(raysONow, raysDNow, boundsNow, timesNow)
    dataNow.attach(managerNow)
    managerTemp.close()
    poseIdx += 1
    pixelIdx = 0
    batchIdx = 0
  }


  override def getData(manager: NDManager): lang.Iterable[Batch] = {
    this.manager = manager
    this
  }

  override def prepare(progress: Progress): Unit = {}

  override def iterator(): util.Iterator[Batch] = {
    poseIdx = 0
    this
  }

  override def hasNext: Boolean = !(poseIdx >= poseNum && batchIdx >= numOfBatch)

  override def next(): Batch = {
    if (managerNow == null) {
      newPixels()
    }
    val subManager = manager.newSubManager()
    val data = new NDList(5)
    val label = new NDList(1)

    val index = if (pixelIdx + pixelStep <= pixelSize) new NDIndex().addSliceDim(pixelIdx, pixelIdx + pixelStep) else new NDIndex().addSliceDim(pixelIdx, pixelSize)
    data.add(dataNow.get(0).get(":"))
    data.add(dataNow.get(1).get(index))
    data.add(dataNow.get(2).get(index))
    data.add(dataNow.get(3).get(":"))

    if (!isRender) {
      data.add(times.get(poseIdxNear(scala.util.Random.nextInt(2))))
      label.add(imageNow.get(index))
    }

    data.attach(subManager)
    label.attach(subManager)

    batchIdx += 1
    if (batchIdx >= numOfBatch) {
      managerNow.close()
      managerNow = null
    } else {
      pixelIdx += pixelStep
    }

    new Batch(subManager, data, label, poseNum, Batchifier.STACK, Batchifier.STACK, batchIdx, numOfBatch)
  }
}

object dNerfDataSet {
  private def getRaysNp(H: Int, W: Int, focal: Float, c2w: NDArray): (NDArray, NDArray) = {
    //c2w：尺寸(3, 4)
    val manager = c2w.getManager
    val w = W * .5f / focal
    val h = H * .5f / focal
    val i = manager.linspace(-w, w, W).broadcast(H, W)
    val j = manager.linspace(-h, h, H).reshape(H, 1).broadcast(H, W)
    val dirs = NDArrays.stack(new NDList(i, j, manager.full(i.getShape, -1, DataType.FLOAT32)), -1)
    val raysD = dirs.matMul(c2w.get(":,:3").transpose(1, 0))
    (c2w.get(":,-1"), raysD.reshape(-1, 3))
    //返回raysO和raysD
  }
}