## Mnist Original 
1. 训练到15轮左右收敛；
2. 在验证集的最好表现在50%左右（lr=0.0001）；
Train Epoch: 22 [53760/54000 (100%)]    Loss: 2.190962
Val set:  Average loss: 2.1564, Accuracy: 3029/6000 (50.48%)
3. 修改lr=1之后在15轮达到收敛，在验证集上面的acc在99.10%;
4. 测试集上面的表现：
（1）Best Model: Test set: Average loss: 0.0280, Accuracy: 9916/10000 (99.16%)
（2）Last Model: Test set: Average loss: 0.0282, Accuracy: 9915/10000 (99.15%)
    修改之后的模型：
 (1) Test set: Average loss: 0.0283, Accuracy: 9905/10000 (99.05%)
 (2) Test set: Average loss: 0.0294, Accuracy: 9902/10000 (99.02%)

## 改进策略 (Debug)
1. 超参数上面进行调整；（重点：学习率的动态调整）
2. 模型架构的调整；（加入残差连接、或者是将层数增加）


## 其余改进
1. 打印日志按照epoch打印（一个epoch只用一行显示进度）-->便于加入保存日志的逻辑；
2. 


## 我进行了的修改
1. 适配优化器将学习率设置为1；
2. 日志打印一行
3. 默认保存模型：val loss最小和最后一轮
