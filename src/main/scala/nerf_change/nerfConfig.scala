package nerf_change

import ai.djl._
import ai.djl.nn._

case class nerfConfig(
                       coarseBlock: Block,
                       fineBlock: Block,
                       /*
                       * 这两个参数是关键的核心参数，分别为粗糙的nerf核心模块和细腻的nerf核心模块，
                       * 这两个模块不能是同一个，至少应该是经过深复制的，
                       * 这两个模块的输入都是一个size为2的NDList，
                       * 其get(0)为origin，get(1)为direction，
                       * 第一个NDArray是三维的，第一维度是batch，第二维度是ray，第三维度是其经过位置编码后的位置，
                       * 第二个NDArray是二维的，第一维度是batch
                       * 这两个模块的输出都是一个size为2的NDList，
                       * 其get(0)为rgb，get(1)为density，
                       * 这两个NDArray都是三维的，第一维度是batch，第二维度是ray，第三维度是3或1
                       */
                       device: Device,
                       /*训练用的设备*/
                       pos_L: Int,
                       direction_L: Int,
                       /*位置和方向的位置编码的阶数*/
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
                       datadir: String,
                       basedir: String
                       /*数据文件夹*/
                     )