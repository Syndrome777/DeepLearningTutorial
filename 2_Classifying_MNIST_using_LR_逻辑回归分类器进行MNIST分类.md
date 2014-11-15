使用逻辑回归分类器进行MNIST分类（Classifying MNIST using Logistic Regressing）
=============================
	本节假定读者属性了下面的Theano概念：[共享变量（shared variable）](http://deeplearning.net/software/theano/tutorial/examples.html#using-shared-variables), [基本数学算子（basic arithmetic ops）](http://deeplearning.net/software/theano/tutorial/adding.html#adding-two-scalars), [Theano的进阶（T.grad）](http://deeplearning.net/software/theano/tutorial/examples.html#computing-gradients), [floatX(默认为float64)](http://deeplearning.net/software/theano/library/config.html#config.floatX)。假如你想要在你的GPU上跑你的代码，你也需要看[GPU](http://deeplearning.net/software/theano/tutorial/using_gpu.html)。
	本节的所有代码可以在[这里](http://deeplearning.net/tutorial/code/logistic_sgd.py)下载。

在这一节，我们将展示Theano如何实现最基本的分类器：逻辑回归分类器。我们以模型的快速入门开始，复习(refresher)和巩固(anchor)数学负号，也展示了数学表达式如何映射到Theano图中。

###模型
逻辑回归模型是一个线性概率模型。它由一个权值矩阵W和偏置向量b参数化。分类通过将输入向量提交到一组超平面，每个超平面对应一个类。输入向量和超平面的距离是这个输入属于该类的一个概率量化。
Theano代码如下。

```Python
		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
```

由于模型的参数需要不断的存取和修正，所以我们把W和b定义为共享变量。这个dot（点乘）和softmax运算用以计算这个P(Y|x,W,b)。这个结果`p_y_given_x`(probability)是一个vector类型的概率向量。
为了获得实际的模型预测，我们使用`T_argmax`操作，来返回`p_y_given_x`的最大值对应的y。
	如果想要获得完整的Theano算子，看[算子列表](http://deeplearning.net/software/theano/library/tensor/basic.html#basic-tensor-functionality)

###定义一个损失函数
学习优化模型参数需要最小化一个损失参数。在多分类的逻辑回归中，很显然是使用负对数似然函数作为损失函数。
虽然整本书都致力于探讨最小化话题，但梯度下降是迄今为止最简单的最小化非线性函数的方法。在这个教程中，我们使用minibatch随机梯度下降算法。可以看[随机梯度下降](http://deeplearning.net/tutorial/gettingstarted.html#opt-sgd)来获得更多细节。
下面的代码定义了一个对给定的minibatch的损失函数。

```Python
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
```
	在这里我们使用错误的平均来表示损失函数，以减少minibatch尺寸对我们的影响。

###创建一个逻辑回归类
现在，我们要定义一个`逻辑回归`的类，来概括逻辑回归的基本行为。代码已经是我们之前涵盖的了，不再进行过多解释。

```Python
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
```
我们通过如下代码来实例化这个类。

```Pyhton
	# generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

```
需要注意的是，输入向量x，和其相关的标签y都是定义在`LogisticRegression`实体外的。这个类需要将输入数据作为`__init__`函数的参数。这在将这些类的实例连接起来构建深网络方面非常有用。一层的输出可以作为下一层的输入。
最后，我们定义了一个`cost`变量来最小化。

```Python
    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)
```

###学习模型
在实现MSGD的许多语言中，需要通过手动求解损失函数对每个参数的梯度（微分）来实现。
在Theano中呢，这是非常简单的。它自动微分，并且使用了一定的数学转换来提高数学稳定性。

```Pyhton
	g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
```
这个函数`train_model`可以被定义如下。
```Python
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
```
`update`是一个list，用以更新每一步的参数。`given`是一个字典，用以表示象征变量，和你在该步中表示的数据。这个`train_model`定义如下：
* 















