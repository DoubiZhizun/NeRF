package nerf

import ai.djl._
import ai.djl.nn._
import ai.djl.training._

case class nerfConfig(
                       device: Device,
                       /*训练用的设备*/
                       posL: Int,
                       dirL: Int,
                       useSH: Boolean,
                       /*位置和方向的位置编码的阶数，如果使用SH则方向不进行位置编码，SH默认三阶*/
                       useTime: Boolean,
                       timeL: Int,
                       /*是否加时间和时间的加入方法，若timeL = 0则加入到网络中，否则以timeL阶傅里叶级数的形式加入
                         实际操作中该值可能跟所表示的时间长度有关*/
                       D: Int,
                       W: Int,
                       skips: Array[Int],
                       /*网络结构，D为深度，W为每层点数，skips为重新插入位置编码结果的地方*/
                       rawNoiseStd: Double,
                       /*混入的噪声的方差，若为0则不混入噪声*/
                       whiteBkgd: Boolean,
                       /*若为true，视背景为白色*/
                       linDisp: Boolean,
                       /*若为true，则采样在视差下为线性，否则在深度下为线性*/
                       NSamples: Int,
                       NImportance: Int,
                       /*粗糙网络的采样点数和细腻网络的额外采样点数，若细腻网络采点数小于0则认为无细腻网络*/
                       perturb: Boolean,
                       /*若为true，给采样加随机位移*/
                       batchNum: Int,
                       /*batchNum*/
                       lrate: Double,
                       lrateDecay: Int,
                       /*学习率与学习率衰减*/
                       ndc: Boolean,
                       /*是否使用ndc变换*/
                       datadir: String,
                       basedir: String
                       /*数据文件夹*/
                     ) {
  var ps: ParameterStore = null
}