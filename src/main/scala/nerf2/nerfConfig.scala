package nerf2

import ai.djl._
import ai.djl.training._

case class nerfConfig(
                       device: Array[Device], //训练用的设备

                       dataSetType: String, //数据集类型，可选llff、blender和deepvoxels

                       //llff数据集选项
                       factor: Int, //llff数据集的下采样率
                       llffHold: Int, //llff数据集中每多少张图片选出一张作为测试集

                       //网络模型配置
                       useDir: Boolean, //是否使用方向参数
                       useSH: Boolean, //是否使用球谐函数
                       useTime: Boolean, //是否使用时间参数
                       useHierarchical: Boolean, //是否使用分层体采样

                       postL: Int, //点与时间的的位置编码阶数
                       dirL: Int, //如果使用方向参数，则该项表示方向的位置编码阶数

                       D: Int, //网络每层宽度
                       W: Int, //网络层数（深度）
                       skips: Array[Int], //网络中再次输入的层

                       NSamples: Int, //粗糙网络采样点数
                       NImportance: Int, //如果使用分层体采样，则该项表示细腻网络的额外采样点数

                       rawNoiseStd: Double, //混入的噪声的方差，若为0则不混入噪声
                       whiteBkgd: Boolean, //若为true，视背景为白色
                       linDisp: Boolean, //若为true，则采样在视差下为线性，否则在深度下为线性
                       perturb: Boolean, //若为true，给采样加随机位移
                       ndc: Boolean, //是否使用ndc变换，llff数据集可能会用到

                       batchNum: Int, //训练每批光线数

                       lrate: Double, //学习率
                       lrateDecay: Int, //学习率衰减，每lrateDecay * 1000个训练周期衰减到原来的0.1倍

                       dataDir: String, //数据文件夹
                       logDir: String, //log文件夹

                       iPrint: Int, //多少次训练周期进行一次打印进度
                       iImage: Int, //多少次训练周期进行一次渲染留档
                       iWeight: Int, //多少次训练周期进行一次权重保存
                       iTestSet: Int, //多少次训练周期进行一次测试机测试
                       iVideo: Int, //多少次训练周期进行一次渲染视频

                       NIter: Int //总训练周期数
                     ) {
  var ps: ParameterStore = null
}