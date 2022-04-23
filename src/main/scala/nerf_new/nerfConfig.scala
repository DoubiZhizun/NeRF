package nerf_new

import ai.djl._
import ai.djl.nn._

case class nerfConfig(
                       device: Device,
                       /*训练用的设备*/
                       pos_L: Int,
                       /*位置的位置编码的阶数*/
                       raw_noise_std: Double,
                       /*混入的噪声的方差，若为0则不混入噪声*/
                       white_bkgd: Boolean,
                       /*若为true，视背景为白色*/
                       lindisp: Boolean,
                       /*若为true，则采样在视差下为线性，否则在深度下为线性*/
                       N_samples: Int,
                       N_importance: Int,
                       /*粗糙网络的采样点数和细腻网络的额外采样点数*/
                       perterb: Boolean,
                       /*若为true，给采样加随机位移*/
                       N_rand: Int,
                       /*batchNum*/
                       lrate: Double,
                       lrate_decay: Int,
                       /*学习率与学习率衰减*/
                       ndc: Boolean,
                       /*是否使用ndc变换*/
                       datadir: String,
                       basedir: String
                       /*数据文件夹*/
                     ) {
  var coarseBlock: nnBlock = null
  var fineBlock: nnBlock = null
  /*
  * 粗糙的nerf核心模块和细腻的nerf核心模块，
  * 这两个模块不能是同一个，至少应该是经过深复制的，
  * 这两个模块的输入都是一个NDArray，其内容为需要渲染的点，尺寸为(batchNum,N_samples,3)
  * 这两个模块的输出都是3个NDArray，
  * 第一个为球谐系数的常数项，尺寸为(batchNum,N_samples,3)
  * 第二个为球谐系数的非常数项，尺寸为(batchNum,N_samples,3,8)
  * 第三个为体密度，尺寸为(batchNum,N_samples-1,1)
  */
}