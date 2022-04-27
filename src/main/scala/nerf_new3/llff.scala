package nerf_new3

//本文件中的函数用于处理llff数据集
//所有的处理过程都使用Float32实现

import ai.djl.ndarray._

object llff {

  def normalize(input: NDArray): NDArray = {
    //归一化坐标系
    input.div(input.norm())
  }

  def recenter(poses: NDArray): NDArray = {
    /*
     * 本函数可将所有相机对准方向的调整为以世界坐标z轴负方向为中心
     * 对于相机方向的相机坐标表示，其z负轴为相机所对准的方向，xy轴平行于近平面
     * xy轴理论上可以任意设定，但实际操作中经常以右为x，上为y
     *
     * poses的最低维度代表不同图片
     * poses高二维的尺寸是3x4，其中3x3的部分是相机坐标在世界坐标中的基坐标，3x1的部分是相机的位置
     *
     * 实现方式：找出相机坐标z轴的均值，作为新坐标系的z轴在世界坐标系的方向
     * 找一个跟它垂直的轴作为x轴，目前为止有两个自由度，可以令z = 0来简化计算
     * 然后根据z轴均值和刚才找出的x找出与他们垂直的y（叉乘）
     * 归一化后得到新的坐标在世界坐标系中的表示
     * 接下来将世界坐标系中的poses变为该坐标系中的poses，然后用该坐标系取代世界坐标系
     */
    val manager = poses.getManager
    val z = normalize(poses.get("...,2").sum(Array(0)))
    //新的z轴
    val (x, y) = getAxis(z)
    //新的x，y轴
    val newWorld = x.getNDArrayInternal.stack(new NDList(y, z), -1)
    manager.create(inv(newWorld.toFloatArray)).matMul(poses)
  }

  def onNear(near: Float, rays_o: NDArray, rays_d: NDArray): NDArray = {
    //将原点重新定位到近端平面上（近端平面坐标：z = -near）
    //返回定位后的rays_o
    val t = rays_o.get("...,2:3").add(near).div(rays_d.get("...,2:3"))
    //用每个o点的z轴距离-1的距离除方向的z轴的值，得出位移量
    rays_o.sub(t.mul(rays_d))
  }

  def ndc(H: Int, W: Int, focal: Float, near: Float, rays_o: NDArray, rays_d: NDArray): NDList = {
    /*
     * 将所有相机的方向都对准世界坐标z轴负方向的目的就是为了在世界坐标下进行ndc变换
     * ndc变换可以将z轴负向无限延伸的台型空间压缩为一个边长为-2的立方体，同时保证原空间中的所有直线依旧都是直线
     * 在z方向上采样时，均匀的采样会被映射到原空间中的根据视差采样
     *
     * H，W：图像的高度和宽度
     * focal：图像的焦距
     * near：近平面在世界坐标中距离原点的位置
     * rays_o：世界坐标中的光线起点
     * rays_d：世界坐标中的光线方向
     *
     * 具体的公式见论文
     */

    val o0 = rays_o.get("...,0").div(rays_o.get("...,2")).mul(-1 / (W / (2 * focal)))
    val o1 = rays_o.get("...,1").div(rays_o.get("...,2")).mul(-1 / (W / (2 * focal)))
    val o2 = NDArrays.div(2 * near, rays_o.get("...,2")).add(1)

    val d0 = rays_d.get("...,0").div(rays_d.get("...,2")).sub(rays_o.get("...,0").div(rays_o.get("...,2"))).mul(-1 / (W / (2 * focal)))
    val d1 = rays_d.get("...,1").div(rays_d.get("...,2")).sub(rays_o.get("...,1").div(rays_o.get("...,2"))).mul(-1 / (W / (2 * focal)))
    val d2 = NDArrays.div(-2 * near, rays_o.get("...,2"))

    new NDList(d0.getNDArrayInternal.stack(new NDList(d1, d2), -1), o0.getNDArrayInternal.stack(new NDList(o1, o2), -1))
  }

  def getAxis(z: NDArray): (NDArray, NDArray) = {
    //输入一个z坐标，得到x和y坐标
    val manager = z.getManager
    val zArray = z.toFloatArray
    val xArray = Array(zArray(1), -zArray(0), 0)
    val xLength = Math.sqrt(zArray(0) * zArray(0) + zArray(1) * zArray(1)).toFloat
    xArray(0) /= xLength
    xArray(1) /= xLength
    val yArray = new Array[Float](3)
    yArray(0) = -zArray(2) * xArray(1)
    yArray(1) = zArray(2) * xArray(0)
    yArray(2) = zArray(0) * xArray(1) - zArray(1) * xArray(0)
    (manager.create(xArray), manager.create(yArray))
  }

  def inv(x: Array[Float]): Array[Array[Float]] = {
    //矩阵求逆
    val xSize = Math.sqrt(x.length).round.toInt
    val y = Array.fill(xSize)(new Array[Float](xSize))
    for (i <- 0 until xSize) {
      val temp = 1 / x(i * xSize + i)
      for (k <- i + 1 until xSize) {
        x(i * xSize + k) *= temp
      }
      for (k <- 0 until i) {
        y(i)(k) *= temp
      }
      y(i)(i) = temp
      for (j <- 0 until xSize) {
        if (i != j) {
          for (k <- i + 1 until xSize) {
            x(j * xSize + k) -= x(j * xSize + i) * x(i * xSize + k)
          }
          for (k <- 0 until i) {
            y(j)(k) -= x(j * xSize + i) * y(i)(k)
          }
          y(j)(i) = -x(j * xSize + i) * temp
        }
      }
    }
    y
  }
}