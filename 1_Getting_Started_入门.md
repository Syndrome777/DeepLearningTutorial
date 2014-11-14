入门（Getting Started）
======================

这个教程并不是为了巩固研究生或者本科生的机器学习课程，但我们确实对一些重要的概念（和公式）做了的快速的概述，来确保我们在谈论同个概念。同时，你也需要去下载数据集，以便可以跑未来课程的样例代码。

###下载
 在每一个学习算法的页面，你都需要去下载相关的文件。加入你想要一次下载所有的文件，你可以克隆本教程的git仓库。

	git clone git://github.com/lisa-lab/DeepLearningTutorials.git

###数据集
#####MNIST数据集
(mnist.pkl.gz)

 [MNIST](http://yann.lecun.com/exdb/mnist)是一个包含60000个训练样例和10000个测试样例的手写数字图像的数据集。在许多论文，包括本教程，都将60000个训练样例分为50000个样例的训练集和10000个样例的验证集（为了超参数，例如学习率、模型尺寸等等）。所有的数字图像都被归一化和中心化为28*28的像素，256位图的灰度图。
 为了方便在Python中的使用，我们对数据集进行了处理。你可以在这里[下载](http://deeplearning.net/data/mnist/mnist.pkl.gz)。这个文件被表示为包含3个lists的tuple：训练集、验证集和测试集。每个lists都是都是两个list的组合，一个list是有numpy的1维array表示的784（28*28）维的0～1（0是黑，1是白）的float值，另一个list是0～9的图像标签。下面的代码显示了如何去加载这个数据集。

	import cPickle, gzip, numpy

	# Load the dataset
	f = gzip.open('mnist.pkl.gz', 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

 当我们使用这个数据集的时候，通常将它分割维几个minibatch。我们建议你将数据集储存为共享变量（shared variables），通过minibatch的索引（一个固定的被告知的batch的尺寸）来存取它们。使用共享变量的原因是为了使用GPU。因为往GPUX显存中复制数据是一个巨大的开销。如果不使用共享变量，GPU代码的运行效率将不会比CPU代码快。如果你将自己的数据定义为共享变量，当共享变量被构建的时候，你就给了Theano在一次请求中将整个数据复制到GPU上的可能。之后，GPU就可以通过共享变量的slice（切片）来存取任何一个minibatch，而不必再从CPU上拷贝数据。同时，因为数据向量（实数）和标签（整数）通常是不同属性的，测试集、验证集和训练集是不同目的的，所以我们建议通过不同的共享变量来储存（这就产生了6个不同的共享变量）。
 由于现在的数据再一个变量里面，一个minibatch被定义为这个变量的一个切片。通过指定它的索引和它的尺寸，可以更加自然的来定义一个minibatch。下面的代码展示了如何去存取数据和如何存取一个minibatch。

    def shared_dataset(data_xy):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets us get around this issue
        return shared_x, T.cast(shared_y, 'int32')

 这个数据以float的形式被存储在GPU上（`dtype`被定义为`theano.confug.floatX`）。然后再将标签转换为int型。
	如果你再GPU上跑代码，并且数据集太大，可能导致内存崩溃。在这个时候，你就应当把数据存储为共享变量。你可以将数据储存为一个充分小的大块（几个minibatch）在一个共享变量里面，然后在训练的时候使用它。一旦你使用了这个大块，更新它储存的值。这将最小化CPU和GPU的内存交换。


###标记
#####数据集标记

#####数学约定

#####符号和缩略语表

#####Python命名空间
	import theano
    import theano.tensor as T
    import numpy


###深度学习的监督优化入门







