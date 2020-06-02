<!--
 * @Descripttion: 
 * @Version: 
 * @Author: jianh
 * @Email: 595495856@qq.com
 * @Date: 2020-06-01 20:45:44
 * @LastEditTime: 2020-06-02 23:03:20
 -->
# HMER
手写公式识别

## 文件
- train.py 训练
- demo_greedy.py 贪婪算法进行推断
- demo_beam.py  集束搜索进行推断
- config.py  配置文件
- model.py 网络模型，包括编码器和解码器
- gen_pkl.py 生成二进制文件
- train.pkl, valid.pkl, test.pkl 分别是训练集，验证集，测试集的二进制文件 

## 文件夹
checkpoints 保存训练权重
data 训练图片和标签文件
result 识别结果txt文件