package nerf_new3

import ai.djl._
import ai.djl.training._

case class nerfConfig(
                       device: Device,
                       /*训练用的设备*/
                       pos_L: Int,
                       /*位置的位置编码的阶数*/
                       raw_noise_std: Double,
                       /*混入的噪声的方差，若为0则不混入噪声*/
                       lindisp: Boolean,
                       /*若为true，则采样在视差下为线性，否则在深度下为线性*/
                       N_samples: Int,
                       /*每条光线采样点数*/
                       perterb: Boolean,
                       /*若为true，给采样加随机位移*/
                       lrate: Double,
                       lrate_decay: Int,
                       /*学习率与学习率衰减*/
                       ndc: Boolean,
                       /*是否使用ndc变换*/
                       Mf: Int,
                       /*生成的特征数量*/
                       factor: Int,
                       /*特征图相对于原图的缩小番数，每大1，特征图缩小一番*/
                       datadir: String,
                       basedir: String
                       /*数据文件夹*/
                     ) {
  var block: nnBlock = null
  /*
  * 模块输入与输出：
  * 输入一项：
  * pos：尺寸(H >> factor, W >> factor, N_samples, 3 + pos_L * 6)
  * 输出一项：
  * fd：尺寸(H, W, N_samples, 28)
  */
  var ps: ParameterStore = null
  //上两个模块的参数仓库
}