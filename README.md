# fashionMNIST_ResNet34

## ResNet34 模型

# ![ResNet34 Model](images/ResNet34.png)

## 模型结构框图

![ResNet34](images/ResNet.jpg)

![](images/resnet-architectures-34-101.png)

## python相关配置

- python 3.6
- modelsummary==1.1.7
- music21==5.7.2
- numpy==1.16.4
- pandas==1.0.4
- tensorboard==1.14.0
- torch==1.4.0+cu100
- torchvision==0.5.0+cu100

## 优化函数

```python
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
```

## 模型准确率

| 训练集准确率            | 验证集准确率            | 测试集准确率            |
| ----------------------- | ----------------------- | ----------------------- |
| <center>88.38%</center> | <center>88.57%</center> | <center>90.26%</center> |

## 模型训练

```python
CUDA_VISIBLE_DEVICE=0 python train.py
```

**CUDA_VISIBLE_DEVICE**对应显卡序号，本实验所用显卡为一块GeForce GTX 1080 Ti：

```python
nvidia-smi
```

训练时会自动保存验证集准确率较高的pth文件，以供测试

## 模型测试

由于本数据集没有区分测试集和验证集，故每次迭代随机从测试集中取batch_size的样本作为验证集，方便训练过程调整超参数

将测试集中的PATH改为对应保存的pth文件即可

```python
CUDA_VISIBLE_DEVICE=0 python test.py
```

## TO DO

- [ ] README.md in English



