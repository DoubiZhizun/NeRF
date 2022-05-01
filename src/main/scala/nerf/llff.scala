package nerf

//本文件中的函数用于处理llff数据集

import ai.djl.modality.cv.util._
import ai.djl.modality.cv._
import ai.djl.ndarray._
import ai.djl.ndarray.index._
import ai.djl.ndarray.types.{DataType, Shape}

import java.io._
import java.nio.file._
import java.util.function._

object llff {

  def normalize(x: NDArray): NDArray = {
    //归一化坐标系
    x.div(x.norm())
  }

  def cross(x: NDArray, y: NDArray): NDArray = {
    //x, y叉积
    val x0 = x.get(0)
    val x1 = x.get(1)
    val x2 = x.get(2)
    val y0 = y.get(0)
    val y1 = y.get(1)
    val y2 = y.get(2)
    val z0 = x1.mul(y2).sub(x2.mul(y1))
    val z1 = x2.mul(y0).sub(x0.mul(y2))
    val z2 = x0.mul(y1).sub(x1.mul(y0))
    z0.getNDArrayInternal.stack(new NDList(z1, z2), -1)
  }

  def inv(x: NDArray): NDArray = {
    //3x3矩阵求逆
    val xEyes = x.concat(x.getManager.eye(3, 3, 0, x.getDataType), -1)
    val xArray = Array(xEyes.get(0), xEyes.get(1), xEyes.get(2))
    val num0 = xArray(0).get(0).getNDArrayInternal.stack(new NDList(xArray(1).get(0), xArray(2).get(0)), 0).argMax().getLong().toInt
    val temp0 = xArray(num0)
    xArray(num0) = xArray(0)
    xArray(0) = temp0
    xArray(0) = xArray(0).div(xArray(0).get(0))
    xArray(1) = xArray(1).sub(xArray(0).mul(xArray(1).get(0)))
    xArray(2) = xArray(2).sub(xArray(0).mul(xArray(2).get(0)))
    val num1 = xArray(2).get(1).stack(xArray(2).get(1), 0).argMax().getLong().toInt
    val temp1 = xArray(num1 + 1)
    xArray(num1 + 1) = xArray(1)
    xArray(1) = temp1
    xArray(1) = xArray(1).div(xArray(1).get(1))
    xArray(0) = xArray(0).sub(xArray(1).mul(xArray(0).get(1)))
    xArray(2) = xArray(2).sub(xArray(1).mul(xArray(2).get(1)))
    xArray(2) = xArray(2).div(xArray(2).get(2))
    xArray(0) = xArray(0).sub(xArray(2).mul(xArray(0).get(2)))
    xArray(1) = xArray(1).sub(xArray(2).mul(xArray(1).get(2)))
    xArray(0).get(new NDIndex().addSliceDim(3, 6)).getNDArrayInternal.stack(new NDList(xArray(1).get(new NDIndex().addSliceDim(3, 6)), xArray(2).get(new NDIndex().addSliceDim(3, 6))), 0)
  }

  def recenter(poses: NDArray): NDArray = {
    /*
     * 本函数可将所有相机对准方向的中心调整为世界坐标z轴负方向，并将所有照相机的位置的均值调整为原点
     * 对于相机方向的相机坐标表示，其z负轴为相机所对准的方向，xy轴平行于近平面
     * xy轴理论上可以任意设定，但实际操作中经常以右为x，上为y
     *
     * poses的最低维度代表不同图片
     * poses高二维的尺寸是3x4，其中3x3的部分是相机坐标在世界坐标中的基坐标，3x1的部分是相机的位置
     *
     * 方向调整实现方式：
     * 找出相机坐标z轴的均值，作为新坐标系的z轴在世界坐标系的方向
     * 找出相机坐标y轴的均值
     * y叉乘z，得到新坐标系的x轴在世界坐标系的方向
     * 此时的x与之前得到的y、z均垂直，但要注意此时的y、z之间并非垂直
     * 所以要用z叉乘x得到新坐标系的y轴在世界坐标系的方向
     * 归一化后得到新的坐标在世界坐标系中的表示
     * 接下来将世界坐标系中的poses变为该坐标系中的poses，然后用该坐标系取代世界坐标系
     * 使用的方法为矩阵除法（即乘逆）
     *
     * 均值调整实现方式：
     * 找出均值并减
     */
    val z = normalize(poses.get("...,2").sum(Array(0)))
    //新的z轴
    val yTemp = poses.get("...,1").sum(Array(0))
    val x = normalize(cross(yTemp, z))
    val y = cross(z, x)
    //新的x，y轴
    val newWorld = x.getNDArrayInternal.stack(new NDList(y, z), -1)
    //新的世界坐标系
    val posesNew = inv(newWorld).matMul(poses)
    //调整位置
    posesNew.set(new NDIndex("...,3"), posesNew.get("...,3").sub(posesNew.get("...,3").mean(Array(0), true)))
    posesNew
  }

  def ndc(H: Int, W: Int, focal: Float, near: Float, raysO: NDArray, raysD: NDArray): (NDArray, NDArray) = {
    /*
     * 将所有相机的方向都对准世界坐标z轴负方向的目的就是为了在世界坐标下进行ndc变换
     * ndc变换可以将z轴负向无限延伸的台型空间压缩为一个边长为-2的立方体，同时保证原空间中的所有直线依旧都是直线
     * 在z方向上采样时，均匀的采样会被映射到原空间中的根据视差采样
     *
     * H，W：图像的高度和宽度
     * focal：图像的焦距
     * near：近平面在世界坐标中到原点的距离（顺带一提，远平面的距离是无穷）
     * raysO：世界坐标中的光线起点
     * raysD：世界坐标中的光线方向
     *
     * 具体的公式见论文
     * 在变换之前会先将原点重新定位到近端平面上（近端平面坐标：z = -near）
     */

    val t = raysO.get("...,2:3").add(near).div(raysD.get("...,2:3"))
    //用每个o点的z轴距离-near的距离除方向的z轴的值，得出位移量
    val raysO2 = raysO.sub(t.mul(raysD))

    val o0 = raysO2.get("...,0").div(raysO2.get("...,2")).mul(-1 / (W / (2 * focal)))
    val o1 = raysO2.get("...,1").div(raysO2.get("...,2")).mul(-1 / (H / (2 * focal)))
    val o2 = NDArrays.div(2 * near, raysO2.get("...,2")).add(1)

    val d0 = raysD.get("...,0").div(raysD.get("...,2")).sub(raysO2.get("...,0").div(raysO2.get("...,2"))).mul(-1 / (W / (2 * focal)))
    val d1 = raysD.get("...,1").div(raysD.get("...,2")).sub(raysO2.get("...,1").div(raysO2.get("...,2"))).mul(-1 / (H / (2 * focal)))
    val d2 = NDArrays.div(-2 * near, raysO2.get("...,2"))

    (o0.getNDArrayInternal.stack(new NDList(o1, o2), -1), d0.getNDArrayInternal.stack(new NDList(d1, d2), -1))
    //返回分别是变换后的raysO和raysD
  }

  def getRenderPoses(rads: NDArray, focal: NDArray, zrate: Float, rots: Int, N: Int): NDArray = {
    /*
     * 根据参数生成渲染所需的方向和原点
     * rads：原点的x、y和z轴的缩小比例，可以取训练集中的原点中的三个方向各自的绝对值最大值的0.9倍
     * focal：对准方向、相机会在原点对准(0, 0, -focal)的方向
     * zrate：z方向旋转速度，即x、y方向上没旋转一周，z方向上旋转的周数
     * rots：x、y方向上的旋转周数
     * N：在rots周上采的点数
     *
     * 渲染出的原点序列是在x、y轴上旋转、在z轴上前后来回平移的，平移与旋转的周期比由zrate来决定
     */
    val manager = rads.getManager
    val thetas = manager.linspace(0, 2 * Math.PI.toFloat * rots, N + 1).get(":-1").toFloatArray
    val renderPoses = new NDList(thetas.length)
    val focalPoint = manager.zeros(new Shape()).toType(focal.getDataType, false).getNDArrayInternal.stack(new NDList(manager.zeros(new Shape()).toType(focal.getDataType, false), focal.neg()), 0)
    val up = manager.create(Array(0, 1, 0)).toType(rads.getDataType, false)
    for (theta <- thetas) {
      val c = manager.create(Array(Math.cos(theta), -math.sin(theta), -math.sin(theta * zrate))).toType(rads.getDataType, false).mul(rads)
      val z = normalize(c.sub(focalPoint))
      val x = cross(up, z)
      val y = cross(z, x)
      renderPoses.add(x.getNDArrayInternal.stack(new NDList(y, z, c), -1))
    }
    NDArrays.stack(renderPoses, 0)
  }

  class int2ArrayPath extends IntFunction[Array[Path]] {
    //辅助函数，用于将Stream[Path]转换为Array[Path]
    override def apply(value: Int): Array[Path] = new Array[Path](value)
  }

  def loadData(datadir: String, factor: Int, manager: NDManager): (NDArray, NDArray, NDArray) = {
    //读取数据，参数含义见loadLlffData
    val posesArr = manager.decode(Files.readAllBytes(Paths.get(datadir, "posesBounds.npy")))
    //posesArr 0维代表不同图片，1维长度为17，分别是大小为3x5的poses和2的bds
    val poses = posesArr.get(":,:-2").reshape(-1, 3, 5)
    val bds = posesArr.get(":,-2:")

    val imageDir = if (factor == 1) Paths.get(datadir, "images") else Paths.get(datadir, s"images_$factor")

    val images = if (Files.exists(imageDir)) {
      //文件夹存在，直接从中读取图片
      val imagesFilePath = Files.list(imageDir).toArray(new int2ArrayPath).sortWith((x, y) => x.compareTo(y) < 0)
      val imagesNDList = new NDList(imagesFilePath.length)
      for (f <- imagesFilePath) {
        imagesNDList.add(ImageFactory.getInstance().fromFile(f).toNDArray(manager).toType(DataType.FLOAT32, false).div(255))
      }
      NDArrays.stack(imagesNDList, 0)
    } else {
      //文件及不存在，创建
      Files.createDirectory(imageDir)
      val imagesFilePath = Files.list(Paths.get(datadir, "images")).toArray(new int2ArrayPath).sortWith((x, y) => x.compareTo(y) < 0)
      val imagesNDList = new NDList(imagesFilePath.length)
      for (f <- imagesFilePath) {
        val image = ImageFactory.getInstance().fromFile(f).toNDArray(manager).toType(DataType.FLOAT32, false)
        var imageResize = NDImageUtils.resize(image, image.getShape.get(1).toInt / factor, image.getShape.get(0).toInt / factor, Image.Interpolation.AREA)
        imageResize = imageResize.minimum(255).maximum(0)
        imagesNDList.add(imageResize.div(255))
        ImageFactory.getInstance().fromNDArray(imageResize.toType(DataType.UINT8, false)).save(new FileOutputStream(Paths.get(imageDir.toString, f.getFileName.toString).toString), "png")
      }
      NDArrays.stack(imagesNDList, 0)
    }

    poses.set(new NDIndex(":,0,4"), images.getShape.get(1))
    poses.set(new NDIndex(":,1,4"), images.getShape.get(2))
    poses.set(new NDIndex(":,2,4"), poses.get(":,2,4").div(factor))

    (poses, bds, images)
    //返回依次为：
    //poses、bds、images
  }

  def loadLlffData(datadir: String, factor: Int = 8, bdFactor: Double = .75, manager: NDManager): (NDArray, NDArray, NDArray, NDArray) = {
    /*
     * 读取llff数据集
     * datadir：放置数据集的文件夹
     * 其中未经压缩的训练/测试集图片放在内部的images文件夹下
     * 经过压缩的图片放在内部的images_n文件夹下，n为压缩倍数（即参数factor）
     * poses和bds数据放在posesBounds.npy中
     * bdFactor：对于边界的压缩倍数，其作用代码里会有详细解释
     */
    var (poses, bds, images) = loadData(datadir, factor, manager)
    //希望读入的数据都将不同图片的索引放在0维

    poses = poses.get("...,1:2").getNDArrayInternal.concat(new NDList(poses.get("...,:1").neg(), poses.get("...,2:")), -1)
    //这句代码的作用是坐标轴转换，原数据集的坐标轴可能是x轴向下，y轴向右，通过令原来的y轴变为现在的x轴，原来的-x轴变为现在的y轴来转换为x轴向右，y轴向上

    val sc = if (bdFactor == 0) manager.create(1) else NDArrays.div(1, bds.min().mul(bdFactor))
    poses.set(new NDIndex(":,:3,3"), poses.get(":,:3,3").mul(sc))
    bds.muli(sc)
    //bds的内容可能是在现实世界中物体上的点到相机的距离
    //如果这个距离过大，后续在将近端平面定为z = -1并进行ndc变换后
    //可能会由于物体距离z = -1过远而被压缩到ndc立方体的很小的一部分空间里
    //并由于采样时分辨率的不足导致图像质量很差
    //所以在实际操作时，会使用相机位置除bds中的最小值与一个因数的积来拉近物体到z = -1的距离
    //理论上讲，拉的越近图像质量越好，但拉的过进可能会导致物体穿过近端平面，所以需要这个因数来调控
    //源代码中这个因数的值为0.75

    poses.set(new NDIndex(":,:3,:4"), recenter(poses.get(":,:3,:4")))
    //将poses重定位

    //准备renderPoses
    val rads = poses.get(":,:3,3").abs().max(Array(0)).mul(.9)

    val closeDepth = bds.min().mul(.9)
    val infDepth = bds.max().mul(5)
    val dt = .75f
    //这三个参数用来确定focal，取视差视角下infDepth到closeDepth中间位于(1 - dt)位置的点
    val focal = NDArrays.div(1, NDArrays.div((1 - dt), closeDepth).add(NDArrays.div(dt, infDepth)))

    val renderPoses = getRenderPoses(rads, focal, .5f, 2, 120)
    //转两圈取120个点

    (poses.toType(DataType.FLOAT32, false), renderPoses.toType(DataType.FLOAT32, false), images, bds.toType(DataType.FLOAT32, false))
    //返回值依次为：
    //poses、renderPoses、images和bds
  }
}