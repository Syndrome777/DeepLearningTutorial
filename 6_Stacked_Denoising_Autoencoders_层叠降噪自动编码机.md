层叠降噪自动编码机（Stacked Denoising Autoencoders (SdA)）
=========================================================

在这一节，我们假设读者已经了解了[使用逻辑回归进行MNIST分类](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/2_Classifying_MNIST_using_LR_逻辑回归进行MNIST分类.md)和[多层感知机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/3_Multilayer_Perceptron_多层感知机.md)。如果你需要在GPU上进行运算，你还需要了解[GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html)。

本节的所有代码可以在[这里](http://deeplearning.net/tutorial/code/SdA.py)下载。

层叠降噪自动编码机（Stacked Denoising Autoencoder，SdA）是层叠自动编码机（[Bengio07](http://deeplearning.net/tutorial/references.html#bengio07)）的一个扩展，在[Vincent08](http://deeplearning.net/tutorial/references.html#vincent08)中被介绍。

这个教程建立在前一个[降噪自动编码机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/5_Denoising_Autoencoders_降噪自动编码.md)之上。我们建议，对于没有自动编码机经验的人应该阅读上述章节。

###层叠自动编码机
降噪自动编码机可以被叠加起来形成一个深度网络，通过反馈前一层的降噪自动编码机的潜在表达（输出编码）作为当前层的输入。这个非监督的预学习结构一次只能学习一个层。每一层都被作为一个降噪自动编码机以最小化重构误差来进行训练。当前k个层被训练完了，我们可以进行k+1层的训练，因此此时我们才可以计算前一层的编码和潜在表达。当所有的层都被训练了，整个网络进行第二阶段训练，称为微调（fine-tuning）。这里，我们考虑监督微调，当我们需要最小化一个监督任务的预测误差吧。为此我们现在网络的顶端添加一个逻辑回归层（使输出层的编码更加精确）。然后我们像训练多层感知器一样训练整个网络。这里，我们考虑每个自动编码的机的编码模块。这个阶段是有监督的，因为我们在训练的时候使用了目标类别（更多细节请看[多层感知机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/3_Multilayer_Perceptron_多层感知机.md)）

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

下一步，我们构建`n_layers`个sigmoid层（我们使用在[多层感知机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/3_Multilayer_Perceptron_多层感知机.md)中介绍的`HiddenLayer`类，唯一的更改是将原本的非线性函数`tanh`换成了logistic函数s=1/(1+exp(-x))）和`n_layers`个降噪自动编码机，当然`n_layers`就是我们模型的深度。我们连接sigmoid函数，使得他们形成一个MLP，构建每一个自动编码机和他们对应的sigmoid层，去共享编码部分的权值矩阵和偏执

```Python
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
```

现在，我们需要在sigmoid层的上方添加逻辑层，所以我们将有一个MLP。我们将使用在[使用逻辑回归进MNIST分类](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/2_Classifying_MNIST_using_LR_逻辑回归进行MNIST分类.md)的`LogisticRegression`类。

```Python
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)
```









