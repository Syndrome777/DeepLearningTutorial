受限波尔兹曼机（Restricted Boltzmann Machines）
==============================================

在这一章节，我们假设读者已经阅读了[使用逻辑回归进行MNIST分类](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/2_Classifying_MNIST_using_LR_逻辑回归进行MNIST分类.md)和[多层感知机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/3_Multilayer_Perceptron_多层感知机.md)。当然，假如你要使用GPU来运行代码，你还需要阅读[GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html)。

本节的所有代码都可以在[这里](http://deeplearning.net/tutorial/code/rbm.py)下载。

###基于能量模型（Energy-Based Models）
基于能量的模型（EBM）把我们所关心变量的各种组合和一个标量能量联系在一起。训练模型的过程就是不断改变标量能量的过程，使其能量函数的形状满足期望的形状。比如，如果一个变量组合被认为是合理的，它同时也具有较小的能量。基于能量的概率模型通过能量函数来定义概率分布：

![energy_fun](/images/7_ebm_1.png)

其中归一化因子Z被称为分割函数：

![Z_fun](/images/7_ebm_2.png)

基于能量的模型可以利用使用梯度下降或随机梯度下降的方法来学习，具体而言，就是以先验（训练集）的负对数似然函数作为损失函数，就像在逻辑回归中我们定义的那样，

![loss_fun](/images/7_ebm_3.png)

其中随机梯度为![gradient](/images/7_ebm_4.png)，其中theta为模型的参数。

####包含隐藏单元的EBMs

在很多情况下，我们无法观察到x样本的全部分布，或者我们需要引进一些没有观察到的变量，以增加模型的表达能力。因而我们考虑将模型分为2部分，一个可见部分（x的观察分布）和一个隐藏部分h，这样得到的就是包含隐含变量的EBM：

![ebm_with_hidden_unit](/images/7_ebm_hidden_units_1.png)

同时我们受物理启发定义了自由能量（free energy）：

![free_energy](/images/7_ebm_hidden_units_2.png)

然后我们可以写成如下公式：

![ebm_with_hidden_units_2](/images/7_ebm_hidden_units_3.png)

数据的服对数似然函数梯度就有如下有趣的形式：

![gradient_rbm_h](/images/7_ebm_hidden_units_4.png)

推倒公式如下：

![gradient_rbm_h_2](/images/7_ebm_hidden_units_5.png)

需要注意的是上述的梯度包含2个项，包括正相位和负相位。正和负的术语不指公式中的每个项的符号，而是反映其对模型所定义的概率密度的影响。第一项增加训练数据的概率（通过减少相关的自由能量），而第二项减小模型产生的样本的概率。

通常我们很难精确计算这个梯度，因为式中第一项涉及到可见单元与隐含单元的联合分布，由于归一化因子Z(θ)的存在，该分布很难获取。 我们只能通过一些采样方法（如Gibbs采样）获取其近似值，其具体方法将在后文中详述。


###受限波尔兹曼机（RBM）

波尔兹曼机是对数线性马尔可夫随机场（MRF）的一种特殊形式，例如这个能量函数在它的自由参数下是线性的。为了使得它们能更强力的表达复杂分布（从受限的参数设定到一个非参数设定），我们认为一些变量是不可见的（被称为隐藏）。通过拥有更多隐藏变量（也称之为隐藏单元），我们可以增加波尔兹曼机的模型容量。受限波尔兹曼机限制波尔兹曼机可视层和隐藏层的层内连接。RBM模型可以由下图描述：

![rbm_graph](/images/7_rbm_1.png)

RBM的能量函数可以被定义如下：

![rbm_energy_fun](/images/7_rbm_2.png)

其中’表示转置，b,c,W为模型的参数，b,c分别为可见层和隐含层的偏置，W为可见层与隐含层的链接权重。

自由能量为如下形式：

![free_energy_rbm](/images/7_rbm_3.png)

由于RBM的特殊结构，可视层和隐藏层层间单元是相互独立的。利用这个特性，我们定义如下：

![prob_rbm](/images/7_rbm_4.png)


























