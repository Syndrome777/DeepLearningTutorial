层叠降噪自动编码机（Stacked Denoising Autoencoders (SdA)）
=========================================================

在这一节，我们假设读者已经了解了[使用逻辑回归进行MNIST分类](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/2_Classifying_MNIST_using_LR_逻辑回归进行MNIST分类.md)和[多层感知机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/3_Multilayer_Perceptron_多层感知机.md)。如果你需要在GPU上进行运算，你还需要了解[GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html)。

本节的所有代码可以在[这里](http://deeplearning.net/tutorial/code/SdA.py)下载。

层叠降噪自动编码机（Stacked Denoising Autoencoder，SdA）是层叠自动编码机（[Bengio](http://deeplearning.net/tutorial/references.html#bengio07)）的一个扩展，在[Vincent08](http://deeplearning.net/tutorial/references.html#vincent08)中被介绍。

这个教程建立在前一个[降噪自动编码机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/5_Denoising_Autoencoders_降噪自动编码.md)。我们建议，对于没有自动编码机经验的人应该阅读上述章节。

###层叠自动编码机
降噪自动编码机可以被叠加起来形成一个深度网络，通过反馈前一层的降噪自动编码机的潜在表达（输出编码）作为当前层的输入。这个非监督的预学习结构一次只能学习一个层。每一层都被作为一个降噪自动编码机以最小化重构误差来进行训练。当前k个层被训练完了，我们可以进行k+1层的训练，因此此时我们才可以计算前一层的编码和潜在表达。当所有的层都被训练了，整个网络进行第二阶段训练，称为微调（fine-tuning）。这里，我们考虑监督微调，当我们需要最小化一个监督任务的预测误差吧。为此我们现在网络的顶端添加一个逻辑回归层（是输出层的编码更加精确）。然后我们像训练多层感知器一样训练整个网络。这里，我们考虑每个自动编码的机的编码模块。这个阶段是有监督的，因为我们在训练的时候使用了目标类别（更多细节请看[多层感知机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/3_Multilayer_Perceptron_多层感知机.md)）

这在Theano里面，使用之前定义的降噪自动编码机，可以轻易的被实现。我们可以将层叠降噪自动编码机看作两部分，一个是自动编码机链表，另一个是一个多层感知机。在预训练阶段，我们使用了第一部分，例如我们将模型看作一系列的自动编码机，然后分别训练每一个自动编码机。在第二阶段，我们使用第二部分。这个两个部分通过分享参数来实现连接。

```Python
class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
```
`self.sigmoid_layers`将会储存多层感知机的sigmoid层，`self.dA_layers`将会储存连接多层感知机层的降噪自动编码机。













