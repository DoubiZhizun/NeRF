package nerf_new

object ArrayFnc {

  //有一些需要用到但不涉及反向传播的函数在这里被定义

  def samplePdf(cdf: Array[Float], z_vals: Array[Float], batchNum: Int, N_samples: Int, N_importance: Int): Array[Float] = {
    //cdf大小：batchNum * (N_samples - 1)
    //z_vals大小：batchNum * N_samples
    //输出大小：batchNum * (N_importance + N_samples)，需要reshape成batchNum * (N_importance + N_samples) * 1的NDArray
    val output = new Array[Float](batchNum * (N_importance + N_samples))
    for (i <- 0 until batchNum) {
      var idx = 0
      var numBefore = 0
      for (j <- 0 until N_samples - 1) {
        output(i * (N_importance + N_samples) + idx) = z_vals(i * N_samples + j)
        idx += 1
        var num = numBefore
        val cdfHere = cdf(i * (N_samples - 1) + j) * (N_importance - 1)
        while (num <= cdfHere) {
          num += 1
        }
        val k = (z_vals(i * N_samples + j + 1) - z_vals(i * N_samples + j)) / (num - numBefore + 1)
        val b = z_vals(i * N_samples + j)
        for (l <- 0 until num - numBefore) {
          output(i * (N_importance + N_samples) + idx + l) = k * (l + 1) + b
        }
        idx += num - numBefore
        numBefore = num
      }
      while (idx < N_importance + N_samples) {
        output(i * (N_importance + N_samples) + idx) = z_vals(i * N_samples + N_samples - 1)
        idx += 1
      }
    }
    output
  }
}