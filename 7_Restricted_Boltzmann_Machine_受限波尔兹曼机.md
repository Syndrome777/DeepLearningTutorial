受限波尔兹曼机（Restricted Boltzmann Machines）
==============================================

在这一章节，我们假设读者已经阅读了[使用逻辑回归进行MNIST分类](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/2_Classifying_MNIST_using_LR_逻辑回归进行MNIST分类.md)和[多层感知机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/3_Multilayer_Perceptron_多层感知机.md)。当然，假如你要使用GPU来运行代码，你还需要阅读[GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html)。

本节的所有代码都可以在[这里](http://deeplearning.net/tutorial/code/rbm.py)下载。

## 基于能量模型（Energy-Based Models）
基于能量的模型（EBM）把我们所关心变量的各种组合和一个标量能量联系在一起。训练模型的过程就是不断改变标量能量的过程，使其能量函数的形状满足期望的形状。比如，如果一个变量组合被认为是合理的，它同时也具有较小的能量。基于能量的概率模型通过能量函数来定义概率分布：

![energy_fun](/images/7_ebm_1.png)

其中归一化因子Z被称为分割函数：

![Z_fun](/images/7_ebm_2.png)

基于能量的模型可以利用使用梯度下降或随机梯度下降的方法来学习，具体而言，就是以先验（训练集）的负对数似然函数作为损失函数，就像在逻辑回归中我们定义的那样，

![loss_fun](/images/7_ebm_3.png)

其中随机梯度为![gradient](/images/7_ebm_4.png)，其中theta为模型的参数。

### 包含隐藏单元的EBMs

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


## 受限波尔兹曼机（RBM）

波尔兹曼机是对数线性马尔可夫随机场（MRF）的一种特殊形式，例如这个能量函数在它的自由参数下是线性的。为了使得它们能更强力的表达复杂分布（从受限的参数设定到一个非参数设定），我们认为一些变量是不可见的（被称为隐藏）。通过拥有更多隐藏变量（也称之为隐藏单元），我们可以增加波尔兹曼机的模型容量。受限波尔兹曼机限制波尔兹曼机可视层和隐藏层的层内连接。RBM模型可以由下图描述：

![rbm_graph](/images/7_rbm_1.png)

RBM的能量函数可以被定义如下：

![rbm_energy_fun](/images/7_rbm_2.png)

其中’表示转置，b,c,W为模型的参数，b,c分别为可见层和隐含层的偏置，W为可见层与隐含层的链接权重。

自由能量为如下形式：

![free_energy_rbm](/images/7_rbm_3.png)

由于RBM的特殊结构，可视层和隐藏层层间单元是相互独立的。利用这个特性，我们定义如下：

![prob_rbm](/images/7_rbm_4.png)


### 二进制单元的RBMs
在使用二进制单元（v和h都属于{0,1}）的普通研究情况时，概率版的普通神经激活函数表示如下：

![activation_fun](/images/7_rbm_binary_units_1.png)

![activation_fun2](/images/7_rbm_binary_units_2.png)

二进制单元RBMs的自由能力为：

![free_energy_binary](images/7_rbm_binary_units_1.png)


### 二进制单元的更新公式

我们可以获得如下的一个二进制单元RBM的对数似然梯度：

![equations](/images/7_update_e_b_u_1.png)

这个公式的更多细节推倒，读者可以阅读[这一页](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DBNEquations)，或者[Learning Deep Architectures for AI](http://www.iro.umontreal.ca/%7Elisa/publications2/index.php/publications/show/239)的第五节。在这里，我们将不使用这些等式，而是通过Theano的`T.grad`来获取梯度。

## 在RBM中进行采样

p(x)的样本可以通过运行马尔可夫链的汇聚、Gibbs采样的过渡来得到。

由N个随机变量（S=(S1,S2,...Sn)）的联合分布的Gibbs采样，可以通过N个采样子步骤来实现,形式如Si～p(Si|S-i)，其中S-i表示集合S中除Si的N-1个随机变量。

我们可以从X的一个任意状态(比如[x1(0),x2(0),…,xK(0)])开始，利用上述条件 分布，迭代的对其分量依次采样，随着采样次数的增加，随机变量[x1(n),x2(n),…,xK(n)]的概率分布将以n的几何级数的速度收敛于X的联合 概率分布P(X)。也就是说，我们可以在未知联合概率分布的条件下对其进行采样。

对于RBMs来说，S包含了可视和隐藏单元的集合。然而，由于它们的条件独立性，可以执行块Gibbs抽样。在这个设定中，可视单元被采样，同时给出隐藏单元的固定值，同样的，隐藏单元也是如此：

![h_v](/images/7_sampling_1.png)

这里，h(n)表示马尔可夫链中第n布的隐藏单元的集合。这意味着，h(n+1)根据概率`simg(W‘v(n)+ci)`来随机地被选为0/1。类似地v(n+1)也是如此。这个过程可以通过下面地图来展现：

![gibbs_sampling](/images/7_sampling_2.png)

当t趋向于无穷时，(v(t),h(t))将越加逼近正确样本的概率分布p(v,h)。

在这个理论里面，每个参数在学习进程中的更新都需要运行这样几个链来趋近。毫无疑问这将耗费很大的计算量。一些新的算法已经被提出来，以有效的学习p(v,h)中的样本情况。


## 对比散度算法（CD-k）

对比散度算法，是一种成功的用于求解对数似然函数关于未知参数梯度的近似的方法。它使用两个技巧来技术采样过程：
* 因为我们希望p(v)=p_train(v)（数据的真实、底层分布），所以我们使用一个训练样本来初始化马尔可夫链（例如，从一个被预计接近于p的分布，所以这个链已经开始去收敛这个最终的分布p）。
* 对比梯度不需要等待链的收敛。样本在k步Gibbs采样后就可以获得。在实际中，k=1时就可以获得惊人的好的效果。


### 持续的对比散度

持续的对比散度[Tieleman08](http://deeplearning.net/tutorial/references.html#tieleman08)使用了另外一种近似方法来从p(v,h)中采样。它建立在一个拥有持续状态的单马尔可夫链上（例如，不是对每个可视样例都重启链）。对每一次参数更新，我们通过简单的运行这个链k步来获得新的样本。然后保存链的状态以便后续的更新。

一般直觉的是，如果参数的更新是足够小相比链的混合率，那么马尔科夫链应该能够“赶上”模型的变化。

## 实现

![RBM_impl](/images/7_implementation_1.png)

我们构建一个RBM类。这个网络的参数既可以通过构造函数初始化，也可以作为参数进行传入。当RBM被用于构建一个深度网络时，这个选项——权重矩阵和隐藏层偏置与一个MLP网络的相应的S形层共享，就是非常有用的。

```Python
class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None
    ):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]
```
下一步就是去定义函数来构建S图。代码如下：

```Python
    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
```

```Python
    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]
```

```Python
    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
```

```Python
    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]
```
现在，我们可以使用这些函数来定义一个Gibbs采样步骤的符号图。我们定义如下两个函数：
* `gibbs_vhv`表示从可视单元中开始的Gibbs采样的步骤。我们将可以看到这对于从RBM中采样是非常有用的。
* `gibbs_hvh`表示从隐藏单元中开始Gibbs采样的步骤。这个函数再实现CD和PCD更新中是非常有用的。
代码如下：

```Python
    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]
```

```Python
    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    # start-snippet-2
```

这个类还有一个函数去计算模型的自由能量，以便去计算参数的梯度。注意我们也会返回pre-sigmoid。

```Python
    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term
```
我们随后添加一个`get_cost_update`方法，目的是产生CD-k和PCD-k的更新的象征性梯度。

```Python
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
```
注意`get_cost_update`作为参数被变量化为`persistent`。这允许我们去使用相同的代码来实现CD和PCD。为了使用PCD，`persistent`需要被关联到一个共享变量，它包含前一次迭代的Gibbs链的状态。

假如`persistent`为`None`，则我们使用正相位时产生的隐藏样本来初始化Gibbs链，以此实现CD。当我们已经建立了这个链的开始点的时候，我们就可以计算这个Gibbs链的终点的样本，以及我们需要的去获得梯度的样本。为了获得这些，我们使用Theano提供的`sacn`操作，我们建议读者去阅读这个[链接](http://deeplearning.net/software/theano/library/scan.html)。

```Python
        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
```
当你已经产生了这个链，我们在链的末尾的样例获得负相位的自由能量。注意，这个`chain_end`是模型参数项中的一个的象征性的Theano变量，当我们简单的求解`T.grad`的时候，这个函数将通过Gibbs链来得到这个梯度。这不是我们想要的（它会搞乱我们的梯度），因此我们需要指示`T.grad`，`chain_end`是一个常量。我们通过`T.grad`的`consider_constant`来做这个事情。

```Python
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
```
最后，我们增加由`scan`返回的更新字典（包含了随机状态的`theano_rng`更新规则）来获取参数更新。在PCD例子中，也需要更新包含Gibbs链状态的共享变量。

```Python
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates
```

## 进展跟踪

RBMs的训练是特别困难的。由于归一化函数Z，我们无法在训练的时候估计对数似然函数log(P(x))。因而我们没有直接可以度量超参数优化与否的方法。

而下面的几个选项对用户是有用的。

### 负样本的检查

在训练中获得的负样本是可以可视化的。在训练进程中，我们知道由RBM定义的模型不断逼近真实分布，p_train(x)。负样例就可以视为训练集中的样本。显而易见的，坏的超参数将在这种方式下被丢弃。

### 滤波器的可视化跟踪

由模型训练的滤波器是可以可视化的。我们可以将每个单元的权值以灰度图的方式展示。滤波器应该选出数据中强的特征。对于任意的数据集，这个滤波器都是不确定的。例如，训练MNIST，滤波器就表现的像“stroke”检测器，而训练自然图像的稀疏编码的时候，则像Gabor滤波器。

### 似然估计的替代

此外，更加容易处理的函数可以被用于做似然估计的替代。当我们使用PCD来训练RBM的时候，可以使用伪似然估计来替代。伪似然估计（Pseudo-likeihood,PL）更加简于计算，因为它假设所有的比特都是相互独立的，因此有：

![PL](/images/7_proxies_likelihood_1.png)




## 主循环



## 结果













