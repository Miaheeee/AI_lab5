# 多模态情感分析

该仓库存储了使用图片+文本构建多模态模型分析情感的代码。

## 设置

你可以通过运行以下代码安装本项目所需依赖。

```python
pip install -r requirements.txt
```



## 仓库结构

以下是一些重要文件及其描述。

```python
|-- data # 图片和文本数据
|-- model # 模型代码
    |-- bert_resnet_simple.py # bert和resnet简单拼接模型
    |-- bert_resnet_weight.py # bert和resnet动态加权模型
    |-- bert_densenet_weight.py # bert和densenet动态加权模型
    |-- txt_or_img.py # 消融实验
|-- report # 项目实验报告
|-- main.py # 模型训练及预测过程
|-- prediction # 预测输出文件
|-- requirement.txt # 运行所需依赖
|-- train.txt # 训练数据
|-- test_with_label.txt # 需要预测的文件
```



##  代码运行的流程

运行主函数，并且输入参数即可开始训练，可选参数为：

`--model` ：模型，可选bert_resnet_simple、bert_resnet_weight、bert_densenet_weight

`--image_only`：只输入图片，不输入默认为false，与text_only参数互斥

`--text_only`：只输入文字，不输入默认为false

 `--lr`：初始学习率，不输入默认为1e-5

`--epoch_num`：训练迭代次数，默认为10次

例如：1.在命令行中输入以下代码，即可选用bert+resnet动态加权的模型，学习率为1e-6，迭代20次

```
python main.py --model bert_resnet_weight --lr 1e-6 --epoch_num 20
```

2.在命令行中输入以下代码，即可运行只输入图片的消融实验，学习率为1e-5，迭代10次

```
python main.py --image_only --model bert_resnet_weight
```



### 参考库

本项目并未参考其他库

参考文章：[多模态情感分析简介 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/381805010)
