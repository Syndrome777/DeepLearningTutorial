卷积神经网络（Convolutional Neural Networks LeNet）
==================================================
在这一节假设读者已经阅读了之前的两章[使用逻辑回归进行MNIST分类](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/2_Classifying_MNIST_using_LR_逻辑回归进行MNIST分类.md)和[多层感知机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/3_Multilayer_Perceptron_多层感知机.md)。
如果你想要在GPU上跑样例，你需要一个好的GPU。至少是1GB显存的。当显示器连接在显卡上时，你可能需要更大的显存。
当GPU连接在显示器上时，每次GPU的调用会有几秒钟的限制。这是必须的，因为GPUs不能在计算的时候，同时被用于显示器。如果没有这个限制，屏幕会长时间不动，就像死机一样。这个例子会说中中等质量的GPUs。当GPU不连接显示器时，没有这个时间限制。你可以通过降低batch的大小来出来延时问题。
本节的所有代码，可以在[这里](http://deeplearning.net/tutorial/code/convolutional_mlp.py)下载，还有[3狼月亮图](https://raw.githubusercontent.com/lisa-lab/DeepLearningTutorials/master/doc/images/3wolfmoon.jpg)。

###动机
卷积神经网络是多层感知机的生物灵感变种。从Hubel和Wiesel先前对猫的视觉皮层的研究，我们知道视皮层中含有细胞的复杂分布。这些细胞只对小的视觉子区域敏感，称为`感受野`。这些子区域平铺来覆盖整个视场。这些细胞表现为输入图像空间的局部滤波器，非常适合检测自然图像中的强空间局部相关性。
此外，两类基础细胞类型被定义：`简单细胞`使用它们的感受野，最大限度的响应特定的棱状图案。`复杂细胞`有更大的感受野，可以局部不变的确定图案精确位置。动物视觉皮层是现存的最强大的视觉处理系统，很显然，我们需要去模仿它的行为。因此，许多类神经模型在文献中出现，包括[NeoCognitron](http://deeplearning.net/tutorial/references.html#fukushima)，[HMAX](http://deeplearning.net/tutorial/references.html#serre07)和[LeNet-5](http://deeplearning.net/tutorial/references.html#lecun98)，这是本教程需要着重讲解的。

###稀疏连接
卷积神经网络通过在相邻层的神经元之间实施局部连接模式来检测局部空间相关性。换句话说就是，第m层的隐藏单元的输入来自第m－1层单元的子集，单元拥有空间上的感受野连接。我们可以通过如下的图来表示：

~[sparse_connectivity](/images/4_sparse_con_1.png)

想象一下，第m－1层是输入视网膜。在上图总，第m层的单元有宽度为3的对输入视网膜的感受野，因此它们只连接视网膜层中3个相邻的神经元。第m层的单元与下一层有相似的连接。我们说，感受野连接于下一层的数目也是3，但是感受野连接于输入的则更大（5）。每个单元对视网膜上于自己感受野相异的地方都是不会有响应的。这个结构保证了被学习的滤波器对一个空间局部输入图案有最强的响应。
然而，就像上面展示的，将这些层叠加起来去形成（非线性）滤波器，就可以变得越来越全局化。举例而言，第m＋1层的单元可以编码一个宽度为5的非线性特征。

###权值共享
此外，在CNNs中，每一只滤波器共享同一组权值，这样该滤波器就可以形成一个特征映射（feaature map）。梯度下降算法在小改动后可以学习这种共享参数。这个被共享权值的梯度就是被共享的参数的梯度的简单求和。
复制单元使得特征可以无视其在视觉野中的位置而被检测到。此外，权值共享增加了学习效率，减少了需要被学习的自由参数的数目。这样的设定，使得CNNs在视觉问题上有更好的泛化性。

###细节和注解





###技巧
####超参的选择

#####滤波器的数量























