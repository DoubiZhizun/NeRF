package hashGridNerf

import ai.djl._

case class nerfConfig(
                       device: Array[Device], //训练用的设备

                       //blender数据集选项
                       halfRes: Boolean, //是否将数据集长宽缩小一半
                       testSkip: Int, //测试集跳跃

                       start: Int, //最低分辨率为每个维度1<<start个点
                       layer: Int, //层数，每层每个维度点数增加一倍，即总数变为原来的8倍
                       T: Int, //一层的点数上限，超过这个点数将会变为哈希空间编码形式
                       featureNum: Int, //每层的特征数量
                       NSamples: Int, //采样点数
                       whiteBkgd: Boolean, //若为true，视背景为白色
                       batchNum: Int, //训练每批光线数
                       //isLinearInterpolation: Boolean, //插值方式，true为三线性插值，false为最邻近插值
                       //目前仅支持最邻近插值
                       dirL: Int, //方向的位置编码阶数
                       rawNoiseStd: Double, //混入的噪声的方差，若为0则不混入噪声

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
                     )