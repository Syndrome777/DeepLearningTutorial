多层感知机（Multilayer Perceptron）
==================================

在本节中，假设你已经了解了[使用逻辑回归进行MNIST分类](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/2_Classifying_MNIST_using_LR_逻辑回归进行MNIST分类.md)。同时本节的所有代码可以在[这里](http://deeplearning.net/tutorial/code/mlp.py)下载.
下一个我们将在Theano中使用的结构是单隐层的多层感知机（MLP）。MLP可以被看作一个逻辑回归分类器。这个中间层被称为隐藏层。一个单隐层对于MLP成为通用近似器是有效的。然而在后面，我们将讲述使用多个隐藏层的好处，例如深度学习的前提。这个课程介绍了[MLP，反向误差传导，如何训练MLPs](http://www.iro.umontreal.ca/~pift6266/H10/notes/mlp.html)。

###模型
一个多层感知机（或者说人工神经网络——ANN）,在只有一个隐藏层时可以被表示为如下的图：





















