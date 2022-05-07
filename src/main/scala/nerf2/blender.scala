package nerf2

//本文件中的函数用于处理blender数据集

import ai.djl.Device
import ai.djl.modality.cv.{Image, ImageFactory}
import ai.djl.modality.cv.util.NDImageUtils
import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.ndarray.{NDArray, NDArrays, NDList, NDManager}
import spray.json._

import java.awt.image.BufferedImage
import java.nio.file._
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer

object blender {

  case class jsonDataType(cameraAngleX: Float, frames: Array[(String, Float, Array[Array[Float]])])

  implicit object reader extends JsonReader[jsonDataType] {
    override def read(json: JsValue): jsonDataType = {
      val jsObj = json.asJsObject.getFields("camera_angle_x", "frames")
      val cameraAngleX = jsObj.head.asInstanceOf[JsNumber].value.toFloat
      val frames = new ArrayBuffer[(String, Float, Array[Array[Float]])]
      val framesElements = jsObj.last.asInstanceOf[JsArray].elements
      for (frame <- framesElements) {
        val frameObj = frame.asJsObject.getFields("file_path", "rotation", "transform_matrix")
        val filePath = frameObj.head.asInstanceOf[JsString].value
        val rotation = frameObj(1).asInstanceOf[JsNumber].value.toFloat
        val transformMatrix = frameObj.last.asInstanceOf[JsArray].elements.map(e =>
          e.asInstanceOf[JsArray].elements.map(e2 =>
            e2.asInstanceOf[JsNumber].value.toFloat
          ).toArray
        ).toArray
        frames += Tuple3(filePath, rotation, transformMatrix)
      }
      jsonDataType(cameraAngleX, frames.toArray)
    }
  }

  def getRenderPoses(theta: Float, phi: Float, radius: Float, manager: NDManager): NDArray = {
    //获取渲染视角
    //radius代表该渲染位置到原点的距离
    //相机方向始终指向原点（即-z方向）
    //phi代表该渲染点与原点的连线跟xy平面之间的夹角（向z方向为正）
    //theta代表该渲染点与原点之间的连线在xy平面上的投影与x轴之间的夹角（向y方向为正）
    //相机坐标的x方向始终位于xy平面上（即z = 0）
    val sinPhi = Math.sin(phi).toFloat
    val cosPhi = Math.cos(phi).toFloat
    val sinTheta = Math.sin(theta).toFloat
    val cosTheta = Math.cos(theta).toFloat
    val origin = manager.create(Array(Array(0, 0, 1, radius), Array(1f, 0, 0, 0), Array(0f, 1, 0, 0)))
    //构造原始矩阵[0, 0, 1, radius
    //           1, 0, 0, 0
    //           0, 1, 0, 0]
    //该矩阵描述了一个位于(radius, 0, 0），指向原点方向，x轴在xy平面上的相机
    val phiMatrix = manager.create(Array(Array(cosPhi, 0, -sinPhi), Array(0f, 1, 0), Array(sinPhi, 0, cosPhi)))
    //该矩阵描述了一个xz方向的旋转，用于将原始矩阵抬起，使之与xy平面之间形成phi度的夹角
    val thetaMatrix = manager.create(Array(Array(cosTheta, -sinTheta, 0), Array(sinTheta, cosTheta, 0), Array(0f, 0, 1)))
    //该矩阵描述了一个xy方向的旋转，用于让抬起phi度后的原始矩阵绕z轴旋转theta读
    thetaMatrix.matMul(phiMatrix).matMul(origin)
    //矩阵乘法结合律
  }

  def image2NDArray(image: BufferedImage, manager: NDManager): NDArray = {
    //将image转换为NDArray
    val width = image.getWidth
    val height = image.getHeight
    val channel = if (image.getType == BufferedImage.TYPE_BYTE_GRAY) 1
    else if (image.getType == BufferedImage.TYPE_3BYTE_BGR) 3
    else if (image.getType == BufferedImage.TYPE_4BYTE_ABGR || image.getType == BufferedImage.TYPE_4BYTE_ABGR_PRE) 4
    else 0
    require(channel != 0)
    val bb = manager.allocateDirect(channel * height * width)
    if (channel == 1) {
      val data = new Array[Int](width * height)
      image.getData.getPixels(0, 0, width, height, data)
      for (gray <- data) {
        bb.put(gray.toByte)
      }
    } else if (channel == 3) {
      val pixels = image.getRGB(0, 0, width, height, null, 0, width)
      for (rgb <- pixels) {
        val red = (rgb >> 16) & 0xFF
        val green = (rgb >> 8) & 0xFF
        val blue = rgb & 0xFF
        bb.put(red.toByte)
        bb.put(green.toByte)
        bb.put(blue.toByte)
      }
    } else {
      val pixels = image.getRGB(0, 0, width, height, null, 0, width)
      for (rgb <- pixels) {
        val alpha = (rgb >> 24) & 0xFF
        val red = (rgb >> 16) & 0xFF
        val green = (rgb >> 8) & 0xFF
        val blue = rgb & 0xFF
        bb.put(red.toByte)
        bb.put(green.toByte)
        bb.put(blue.toByte)
        bb.put(alpha.toByte)
      }
    }
    manager.create(bb, new Shape(height, width, channel), DataType.UINT8)
  }

  def loadBlenderData(dataDir: String, halfRes: Boolean = false, testSkip: Int = 1, manager: NDManager): (NDArray, NDArray, NDArray, Array[Float], Array[Int]) = {
    //dataDir：数据路径
    //halfRes：若为true，则将图片长宽变为原来的一半
    //testSkip：测试集跳选
    val splits = Array("train", "val", "test")
    val metas = splits.map(s => JsonParser(Files.readAllBytes(Paths.get(dataDir, s"transforms_$s.json"))).convertTo)
    val allImages = new NDList()
    val allPoses = new NDList()
    var H: Int = 0
    var W: Int = 0
    val iSplit = Array(0, 0, 0)
    for (i <- splits.indices) {
      val meta = metas(i)
      val skip = if (splits(i) == "train" || testSkip == 0) 1 else testSkip
      val indices = 0 until(meta.frames.length, skip)
      val images = new NDList(indices.length)
      val poses = new NDList(indices.length)
      for (i <- indices) {
        val image = ImageIO.read(Paths.get(dataDir, meta.frames(i)._1 + ".png").toFile)
        H = image.getHeight
        W = image.getWidth
        images.add((if (halfRes) NDImageUtils.resize(image2NDArray(image,manager), W / 2, H / 2, Image.Interpolation.AREA) else image2NDArray(image,manager)).toType(DataType.FLOAT32, false).div(255))
        poses.add(manager.create(meta.frames(i)._3).get(":3"))
      }
      allImages.addAll(images)
      allPoses.addAll(poses)
      iSplit(i) = allImages.size()
    }
    if (halfRes) {
      H /= 2
      W /= 2
    }
    val focal =.5f * W / Math.tan(.5 * metas.head.cameraAngleX).toFloat
    val images = NDArrays.stack(allImages, 0)
    val poses = NDArrays.stack(allPoses, 0)
    val renderPoses = NDArrays.stack(new NDList((0 until 40).map(i => getRenderPoses(2 * Math.PI.toFloat / 40 * i, Math.PI.toFloat / 6, 4, manager)).toArray: _*), 0)
    (poses, renderPoses, images, Array(H, W, focal), iSplit)
  }
}