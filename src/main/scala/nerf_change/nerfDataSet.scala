package nerf_change

import ai.djl.ndarray._
import ai.djl.ndarray.index.NDIndex
import ai.djl.training.dataset._
import ai.djl.util.Progress

import java.{lang, util}

final class nerfDataSet(rays_o: NDArray, rays_d: NDArray, bounds: NDArray, viewdirs: NDArray, images: NDArray, batchNum: Int) extends Dataset with java.lang.Iterable[Batch] with java.util.Iterator[Batch] {

  val manager = rays_o.getManager
  val size = rays_o.getShape.get(0).toInt
  var now = 0
  var idx = 0
  val totalNum = Math.ceil(size.toDouble / batchNum).toInt
  val all = new NDList(rays_o, rays_d, bounds, viewdirs, images)


  override def getData(manager: NDManager): lang.Iterable[Batch] = this

  override def prepare(progress: Progress): Unit = {}

  override def iterator(): util.Iterator[Batch] = {
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
    for (i <- 0 until 4) {
      data.add(all.get(i).get(index))
    }
    label.add(all.get(4).get(index))
    data.attach(subManager)
    label.attach(subManager)
    now += batchNum
    idx += 1
    new Batch(subManager, data, label, 1, null, null, idx - 1, totalNum)
  }
}