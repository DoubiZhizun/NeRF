package nerf

import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.training.dataset._
import ai.djl.translate.Batchifier
import ai.djl.util.Progress

import java._

final class nerfDataSet(data: NDList, label: NDList, batchNum: Int) extends Dataset with java.lang.Iterable[Batch] with java.util.Iterator[Batch] {

  val manager = data.get(0).getManager
  val size = data.get(0).getShape.get(0).toInt
  var now = 0
  var idx = 0
  val totalNum = Math.ceil(size.toDouble / batchNum).toInt
  var allData: NDList = null
  var allLabel: NDList = null


  override def getData(manager: NDManager): lang.Iterable[Batch] = this

  override def prepare(progress: Progress): Unit = {}

  override def iterator(): util.Iterator[Batch] = {
    if (allData != null) {
      allData.close()
    }
    if (allLabel != null) {
      allLabel.close()
    }
    val list = (0 until size).toArray
    for (i <- 0 until size) {
      val j = scala.util.Random.nextInt(size)
      val temp = list(j)
      list(j) = list(i)
      list(i) = temp
    }
    allData = new NDList(data.size())
    allLabel = new NDList(label.size())
    val subManager = manager.newSubManager()
    val ndArrayList = subManager.create(list).expandDims(-1)
    for (i <- 0 until data.size()) {
      allData.add(data.get(i).get(new NDIndex().addPickDim(ndArrayList.repeat(data.get(i).getShape))))
    }
    for (i <- 0 until label.size()) {
      allLabel.add(label.get(i).get(new NDIndex().addPickDim(ndArrayList.repeat(data.get(i).getShape))))
    }
    subManager.close()
    now = 0
    idx = 0
    this
  }

  override def hasNext: Boolean = idx < totalNum

  override def next(): Batch = {
    val subManager = manager.newSubManager()
    val data = new NDList(allData.size())
    val label = new NDList(allLabel.size())
    val index = if (now + batchNum > size) new NDIndex().addSliceDim(now, size) else new NDIndex().addSliceDim(now, now + batchNum)
    for (i <- 0 until allData.size()) {
      data.add(allData.get(i).get(index))
    }
    for (i <- 0 until allLabel.size()) {
      label.add(allLabel.get(i).get(index))
    }
    data.attach(subManager)
    label.attach(subManager)
    now += batchNum
    idx += 1
    new Batch(subManager, data, label, 1, Batchifier.STACK, Batchifier.STACK, idx - 1, totalNum)
  }
}