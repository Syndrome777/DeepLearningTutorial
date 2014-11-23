层叠降噪自动编码机（Stacked Denoising Autoencoders (SdA)）
=========================================================

在这一节，我们假设读者已经了解了[使用逻辑回归进行MNIST分类](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/2_Classifying_MNIST_using_LR_逻辑回归进行MNIST分类.md)和[多层感知机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/3_Multilayer_Perceptron_多层感知机.md)。如果你需要在GPU上进行运算，你还需要了解[GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html)。

本节的所有代码可以在[这里](http://deeplearning.net/tutorial/code/SdA.py)下载。

层叠降噪自动编码机（Stacked Denoising Autoencoder，SdA）是层叠自动编码机（[Bengio07](http://deeplearning.net/tutorial/references.html#bengio07)）的一个扩展，在[Vincent08](http://deeplearning.net/tutorial/references.html#vincent08)中被介绍。

这个教程建立在前一个[降噪自动编码机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/5_Denoising_Autoencoders_降噪自动编码.md)之上。我们建议，对于没有自动编码机经验的人应该阅读上述章节。

###层叠自动编码机
降噪自动编码机可以被叠加起来形成一个深度网络，通过反馈前一层的降噪自动编码机的潜在表达（输出编码）作为当前层的输入。这个非监督的预学习结构一次只能学习一个层。每一层都被作为一个降噪自动编码机以最小化重构误差来进行训练。当前k个层被训练完了，我们可以进行k+1层的训练，因此此时我们才可以计算前一层的编码和潜在表达。当所有的层都被训练了，整个网络进行第二阶段训练，称为微调（fine-tuning）。这里，我们考虑监督微调，当我们需要最小化一个监督任务的预测误差吧。为此我们现在网络的顶端添加一个逻辑回归层（使输出层的编码更加精确）。然后我们像训练多层感知器一样训练整个网络。这里，我们考虑每个自动编码的机的编码模块。这个阶段是有监督的，因为我们在训练的时候使用了目标类别（更多细节请看[多层感知机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/3_Multilayer_Perceptron_多层感知机.md)）

![SdA](/images/6_sda_1.png)

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
这个类也提供一个方法去产生与不同层相关的降噪自动编码机的训练函数。它们以list的形式返回，第i个元素就是一个实现训练第i层的`dA`的函数。

```Python
    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
```
为了有能力在训练时，改变差错等级或者训练速率。我们用一个Theano变量来联系它们。

```Python
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
```
现在任何一个`pretrain_fns[i]`函数，可以将`index`，`corruption`——差错等级，`lr`——学习速率作为参数。注意，这些参数的名字是Theano变量的名字，而不是Python变量的名字（`learning_rate`或者`corruption_level`）。在使用Theano时，注意这一点。

以相同的方式（fashion），我们创建了一个方法用于在微调（fine-tuning）时需要的构建函数（`train_model`，`validate_model`，`test_model`函数）。

```Python
    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score
```
注意，这里返回的`valid_score`和`test_score`并不是Theano函数，而是Python函数，在整个验证集和测试集循环，以产生这些集合的损失数的list。

###将它组合起来

下面的几行代码去构建层叠自动编码机：
```Python
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=28 * 28,
        hidden_layers_sizes=[1000, 1000, 1000],
        n_outs=10
    )
```
在训练这个网络时，有两个阶段，一层是预训练，之后是微调。

对于预训练阶段，我们将循环网络中的所有层。对于每一层，我们将使用编译的theano函数来实现SGD(随机梯度下降)，以实现权值优化，来见效每层的重构损失。这个函数将在训练集中被应用，并且是以`pretraining_epochs`中给出的固定次数的迭代。

```Python
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [.1, .2, .3]
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
```
这个微调（fine-tuning）循环和[多层感知机](https://github.com/Syndrome777/DeepLearningTutorial/blob/master/3_Multilayer_Perceptron_多层感知机.md)中的非常类似，唯一的不同是我们将使用在`build_funetune_functions`中给定的新函数。

###运行这个代码
默认情况下，这个代码以块数目为1，每一层循环15次来进行预训练预训练。错差等级（corruption level）在第一层被设为0.1，第二层被设为0.2，第三层被设为0.3。预训练的学习速率为0.001，微调学习速率为0.1。预训练花了585.01分钟，平均每层13分钟。微调在36次迭代，444.2分钟后完成。平均每层迭代12.34分钟。最后的验证得分为1.39%，测试得分为1.3%。所有的结果都是在Intel Xeon E5430 @ 2.66GHz CPU，GotoBLAS下得出。

###技巧
这里有一个方法去提高代码的运行速度（假定你有足够的可用内存），是去计算这个网络（直到第k-1层时）如何转换你的数据。换句话说，你通过训练你的第一个dA层来开始。一旦它被训练，你就可以为每一个数据节点计算隐单元的值然后将它们储存为一个新的数据集，以便你在第2层中训练dA。一旦你训练完第2层的dA，你以相同的方式计算第三层的数据。现在你可以明白，在这个时候，这个dAs被分开训练了，它们仅仅提供（一对一的）对输入的非线性转换。一旦所有的dAs被训练，你就可以开始微调整个模型了。















