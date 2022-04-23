package nerf_new

import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.training.dataset._
import ai.djl.util.Progress

import java._

final class nerfDataSet(rays_o: NDArray, rays_d: NDArray, near: NDArray, far: NDArray, viewdirs: NDArray, images: NDArray, batchNum: Int) extends Dataset with java.lang.Iterable[Batch] with java.util.Iterator[Batch] {

  val manager = rays_o.getManager
  val size = rays_o.getShape.get(0).toInt
  var now = 0
  var idx = 0
  val totalNum = Math.ceil(size.toDouble / batchNum).toInt
  var all: NDList = null


  override def getData(manager: NDManager): lang.Iterable[Batch] = this

  override def prepare(progress: Progress): Unit = {}

  override def iterator(): util.Iterator[Batch] = {
    if (all != null) {
      all.close()
    }
    val subManager = manager.newSubManager()
    val list = (0 until size).toArray
    for (i <- 0 until size) {
      val j = scala.util.Random.nextInt(size)
      val temp = list(j)
      list(j) = list(i)
      list(i) = temp
    }
    var ndArrayList = subManager.create(list).expandDims(-1)
    val nearNew = near.get(new NDIndex().addPickDim(ndArrayList))
    val farNew = far.get(new NDIndex().addPickDim(ndArrayList))
    ndArrayList = ndArrayList.broadcast(size, 3)
    val imagesNew = images.get(new NDIndex().addPickDim(ndArrayList))
    ndArrayList = ndArrayList.expandDims(-2)
    val rays_oNew = rays_o.get(new NDIndex().addPickDim(ndArrayList))
    val rays_dNew = rays_d.get(new NDIndex().addPickDim(ndArrayList))
    ndArrayList = ndArrayList.expandDims(-2)
    val viewdirsNew = viewdirs.get(new NDIndex().addPickDim(ndArrayList))
    subManager.close()
    all = new NDList(rays_oNew, rays_dNew, nearNew, farNew, viewdirsNew, imagesNew)
    now = 0
    idx = 0
    this
  }

  override def hasNext: Boolean = idx < totalNum

  override def next(): Batch = {
    val subManager = manager.newSubManager()
    val data = new NDList(4)
    val label = new NDList(1)
    val index = if (now + batchNum > size) new NDIndex().addSliceDim(now, size) else new NDIndex().addSliceDim(now, now + batchNum)
    for (i <- 0 until 5) {
      data.add(all.get(i).get(index))
    }
    label.add(all.get(5).get(index))
    data.attach(subManager)
    label.attach(subManager)
    now += batchNum
    idx += 1
    new Batch(subManager, data, label, 1, null, null, idx - 1, totalNum)
  }
}