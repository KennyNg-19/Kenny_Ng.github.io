---
title: datawhale-cv训练营-02数据读取与数据扩增
date: 2020-05-23 12:39:17
tags: [ML, CV]
---



在上一章节，官方提供了三种不同的解决方案。从本章开始 将逐渐的学习使用**【定长字符识别】思路**来构建模型，讲解赛题的解决方案和相应知识点。

## 2 数据读取与数据扩增

### 2.1 学习目标

本章主要内容为**图像数据读取、数据扩增方法**和**实战Pytorch读取赛题数据**三个部分组成。

- 学习Python和Pytorch中图像读取
- 学会扩增方法和实战Pytorch读取赛题数据

### 2.2 图像读取

在识别之前，首先需要完成**对数据的读取操作**。在Python中有很多库可以完成数据读取的操作，比较常见的有**Pillow和OpenCV**。



#### 2.2.1 Pillow

Pillow是Python图像**处理函式库(PIL）**的一个分支。Pillow提供了常见的图像读取和处理的操作。



#### 2.2.2 OpenCV

OpenCV是一个跨平台的**计算机视觉库**。OpenCV发展的非常早，拥有**众多的计算机视觉、数字图像处理和机器视觉等功能**。OpenCV在功能上<font color="#dd0000">**比Pillow更加强大很多，但学习成本也高很多**</font>。





### 2.3 数据扩增

现在回到赛题街道字符识别任务中。在赛题中我们需要对的图像进行字符识别，因此需要我们完成的数据的读取操作，同时也需要完成**数据扩增（Data Augmentation）操作**。

#### 2.3.1 基本介绍

在深度学习中数据扩增方法非常重要，数据扩增可以增加训练集的样本，同时也可以有效缓解模型过拟合的情况，也可以给模型带来的更强的泛化能力。



已知，在深度学习模型的训练过程中，数据扩增是<font color="#dd0000">必不可少的环节</font>。

- #### <font color="#dd0000">数据扩增为什么有用？</font>

1. 现有深度学习的参数非常多，一般的模型可训练的<font color="#dd0000">**参数量基本上都是万到百万级别，而训练集样本的数量很难有这么多**</font>。

2. 其次数据扩增可以<font color="#dd0000"> **扩展样本空间**</font>：假设现在的分类模型需要对汽车进行分类

   如果**不使用任何数据扩增方法**，深度学习模型会**从汽车车头的角度❌**来进行判别，**而不是汽车具体的区别✅**。

![汽车分类](https://github.com/datawhalechina/team-learning/raw/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5%EF%BC%88%E8%A1%97%E6%99%AF%E5%AD%97%E7%AC%A6%E7%BC%96%E7%A0%81%E8%AF%86%E5%88%AB%EF%BC%89/IMG/Task02/%E6%95%B0%E6%8D%AE%E6%89%A9%E5%A2%9Ecar.png)



#### 2.3.2 常见的数据扩增方法

- #### 有哪些数据扩增方法？

数据扩增方法有很多：<font color="#dd0000">从**颜色空间、尺度空间到样本空间**，同时根据不同任务数据扩增都有相应的区别</font>。

> 对于图像分类，数据扩增一般<font color="#dd0000">**不会改变标签**</font>;(即本比赛的**需求**场景)
>
> 对于物体检测，数据扩增会改变物体坐标位置；
>
> 对于图像分割，数据扩增会改变像素标签。



**Note**: 在本次赛题中，赛题任务是需要对图像中的字符进行识别，因此对于**<font color="#dd0000">字符图片并不能进行翻转操作</font>。比如字符6经过水平翻转就变成了字符9**，<font color="#dd0000">**会改变字符原本的含义**</font>



- #### **具体**常用的方法和数据扩增<u>库</u>

在常见的数据扩增方法中，一般会从**图像颜色、尺寸、形态、空间和像素等角度**进行变换。当然不同的数据扩增方法可以自由进行组合，得到更加丰富的数据扩增方法。

以**[torchvision](https://pytorch.org/docs/stable/torchvision/index.html)**(pytorch官方提供的数据扩增库，提供了基本的数据数据扩增方法，可以<u>无缝与torch进行集成</u>；但<u>数据扩增方法种类较少，且速度中等</u>)为例，常见的数据扩增方法（API）包括：

```python
- transforms.CenterCrop 对图片中心进行裁剪
- transforms.ColorJitter 对图像颜色的对比度、饱和度和零度进行变换
- transforms.FiveCrop 对图像四个角和中心进行裁剪得到五分图像
- transforms.Grayscale 对图像进行灰度变换
- transforms.Pad 使用固定值进行像素填充
- transforms.RandomAffine 随机仿射变换
- transforms.RandomCrop 随机区域裁剪
- transforms.RandomHorizontalFlip 随机水平翻转
- transforms.RandomRotation 随机旋转
- transforms.RandomVerticalFlip 随机垂直翻转
```



除了torchvision，还有速度**更快的第三方扩增库**供选择：

1. [imgaug](https://github.com/aleju/imgaug) 提供了多样的数据扩增方法，且组合起来非常方便，速度较快；

2. [albumentations](https://albumentations.readthedocs.io) 提供了多样的数据扩增方法，对图像分类、语义分割、物体检测和关键点检测都支持，速度较快。



## 2.4 Pytorch读取数据

由于本次赛题我们使用Pytorch框架讲解具体的解决方案，接下来将是解决赛题的**第一步**使用**Pytorch读取赛题数据**。

#### 2.4.1 一些定义(写代码前想好大致逻辑)

首先，**区分**Dataset和DataLoder这两个<u>数据处理的常用术语</u> 和 **解释**有了Dataset为什么还要有DataLoder？

其实这两个是两个不同的概念，是为了**实现不同的功能**。

- Dataset：对<font color="#dd0000">**数据集的封装**</font>，提供索引方式的对数据样本进行读取
- DataLoder：对<font color="#dd0000">**Dataset进行封装**</font>，提供批量读取的迭代读取

而<font color="#dd0000">**在Pytorch中的数据读取逻辑**</font>， 数据①先**通过Dataset进行封装**，②再<u>**通过DataLoder进行并行读取**</u>。



#### 2.4.2 代码

##### Step①定义对数据集封装的Dataset(详情，见注释)

```python
# 省略各种import 
class SVHNDataset(Dataset):
  	# constructor
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label 
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
		
    # getter: 因为Dataset是提供【索引方式】的对数据样本进行读取
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl)  + (5 - len(lbl)) * [10]
        
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)
      
# 获取: 图片数据和label的路径
train_path = glob.glob('../input/train/*.png')
train_path.sort()

train_json = json.load(open('../input/train.json'))
train_label = [train_json[x]['label'] for x in train_json]

# 定义数据集实例
data = SVHNDataset(train_path, train_label,
          transforms.Compose([
              # 缩放到固定尺寸
              transforms.Resize((64, 128)),
							
            	########################## 数据扩增 ##########################
              # 随机颜色变换
              transforms.ColorJitter(0.2, 0.2, 0.2),
              # 加入随机旋转
              transforms.RandomRotation(5),

              # 将图片转换为pytorch 的tesntor
              transforms.ToTensor(),
              # 对图像像素进行归一化
							transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]))
```



##### Step②定义对Dataset封装的DataLoader

加入DataLoder：数据按照**批次(batch_size=10)**获取，每批次调用Dataset读取单个样本进行拼接。



此时data的格式为：`torch.Size([10, 3, 64, 128]), torch.Size([10, 6])`。

前者为图像文件，为batchsize * chanel * height * width次序；后者为字符标签。

```python
train_loader = torch.utils.data.DataLoader(data, # 封装上面的dataset即可
    batch_size=10, # 每批样本个数
    shuffle=False, # 是否打乱顺序
    num_workers=10, # 读取的线程个数
)

for data in train_loader:
		# 后续操作...
```

