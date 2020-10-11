# 多图像拼接代码实现

## 实验环境：

```python
opencv-python = 3.4.2.16
opencv-contrib-python = 3.4.2.16
matplotlib
numpy
```

## 输入和输出

1.需要拼接的图像按从左到右的顺序分别命名为img1,img2,img3,...

2.手机拍摄的图像像素较大，一轮迭代可能需要4-6分钟，建议使用截图工具。

3.最终输出的图像是bestpanorma2.png（2是迭代轮数，你输入的图片数量减1）。

## 实验结果说明

#### Task1

**这个task的拼接缝不明显，用于说明多图像拼接的流程，示例为三张图像的拼接，进行两轮迭代**。

1. 经过reshape大小统一和亮度平均后的原始输入（前两张）

<img src="/images/imgsOFreadme/p1.png" style="zoom: 80%;" />

2. 第一轮的匹配点连接图，对应matching1.png

![](/images/imgsOFreadme/p2.png)

3. 第一轮简单拼接后的图像(simplepanorma1.png)，去除拼接缝后的图像(bestpanorma1.png)下面一张去除了无效区域和拼接缝。

<img src="/images/imgsOFreadme/p3.png" style="zoom:80%;" />

4. 第二轮经过reshape大小统一和亮度平均后的原始输入（第一轮迭代的输出和第三张图片）

   <img src="/images/imgsOFreadme/p4.png" style="zoom:80%;" />

5. 第二轮的匹配点连接图，对应matching1.png

   ![](/images/imgsOFreadme/p5.png)

6. 第二轮简单拼接后的图像(implepanorma1.png)，去除拼接缝后的图像(bestpanorma1.png)

   <img src="/images/imgsOFreadme/p6.png" style="zoom:80%;" />

#### Task2

**拼接缝明显的两图像拼接（使用参考博客图片）**

原图：

<img src="/images/imgsOFreadme/p7.png" style="zoom:80%;" />

匹配点连线：

<img src="/images/imgsOFreadme/p8.png" style="zoom:80%;" />

简单拼接&去除拼接缝：

<img src="/images/imgsOFreadme/p9.png" style="zoom:80%;" />

## 参考博客

https://blog.csdn.net/qq_37734256/article/details/86745451