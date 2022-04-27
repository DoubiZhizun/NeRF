package nerf_new3

import ai.djl.ndarray._
import ai.djl.training.dataset._
import ai.djl.util.Progress

import java._

final class nerfDataSet(c2w: NDArray, rays_o: NDArray, images: NDArray) extends Dataset with java.lang.Iterable[Batch] with java.util.Iterator[Batch] {

  val manager = rays_o.getManager
  val size = rays_o.getShape.get(0).toInt
  var idx = 0
  val sort = (0 until size).toArray


  override def getData(manager: NDManager): lang.Iterable[Batch] = this

  override def prepare(progress: Progress): Unit = {}

  override def iterator(): util.Iterator[Batch] = {
    for (i <- 0 until size) {
      val j = scala.util.Random.nextInt(size)
      val temp = sort(j)
      sort(j) = sort(i)
      sort(i) = temp
    }
    idx = 0
    this
  }

  override def hasNext: Boolean = idx < size

  override def next(): Batch = {
    val subManager = manager.newSubManager()
    val data = new NDList(c2w.get(sort(idx)), rays_o.get(sort(idx)))
    val label = new NDList(images.get(sort(idx)))
    data.attach(subManager)
    label.attach(subManager)
    idx += 1
    new Batch(subManager, data, label, 1, null, null, idx - 1, size)
  }
}