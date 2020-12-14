# coding:utf-8
# author: xi'an Li
# date: 2020.12.12
# 导入本次需要的模块
import tensorflow as tf
import numpy as np


def mean_var2tensor(input_variable):
    v_shape = input_variable.get_shape()
    axis = [len(v_shape) - 1]
    v_mean, v_var = tf.nn.moments(input_variable, axes=axis, keep_dims=True)
    return v_mean, v_var


def mean_var2numpy(input_variable):
    v_shape = input_variable.get_shape()
    axis = [len(v_shape) - 1]
    v_mean, v_var = tf.nn.moments(input_variable, axes=axis, keep_dims=True)
    return v_mean, v_var


def my_batch_normalization(input_x, is_training=True, name='BatchNorm', moving_decay=0.9):
    # Batch Normalize
    x_shape = input_x.get_shape()
    axis = [len(x_shape) - 1]
    with tf.variable_scope(name):
        x_mean, x_var = tf.nn.moments(input_x, axes=axis, name='moments', keep_dims=True)
        scale = tf.constant(0.1)  # 所有的batch 使用同一个scale因子
        shift = tf.constant(0.001)  # 所有的batch 使用同一个shift项
        epsilon = 0.0001

        # 采用滑动平均更新均值与方差
        ema = tf.train.ExponentialMovingAverage(moving_decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([x_mean, x_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(x_mean), tf.identity(x_var)

        # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
        x_mean, x_var = tf.cond(tf.equal(is_training, True), mean_var_with_update,
                                lambda: (ema.average(x_mean), ema.average(x_var)))

        out_x = tf.nn.batch_normalization(input_x, x_mean, x_var, shift, scale, epsilon)
        return out_x


def my_bn(input_x, is_training=True, name='BatchNorm', moving_decay=0.9):
    # Batch Normalize
    x_shape = input_x.get_shape()
    axis = [len(x_shape) - 1]
    with tf.variable_scope(name):
        x_mean, x_var = tf.nn.moments(input_x, axes=axis, name='moments', keep_dims=True)
        scale = tf.constant(0.1)  # 所有的batch 使用同一个scale因子
        shift = tf.constant(0.001)  # 所有的batch 使用同一个shift项
        epsilon = 0.0001
        out_x = tf.nn.batch_normalization(input_x, x_mean, x_var, shift, scale, epsilon)
        return out_x


# ---------------------------------------------- my activations -----------------------------------------------
def mysin(x):
    return tf.sin(2*np.pi*x)


def srelu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)


def s2relu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)


def stanh(x):
    # return tf.tanh(x)*tf.sin(2*np.pi*x)
    return tf.sin(2*np.pi*tf.tanh(x))


def gauss(x):
    return tf.exp(-0.5*x*x)


def mexican(x):
    return (1-x*x)*tf.exp(-0.5*x*x)


def modify_mexican(x):
    # return 1.25*x*tf.exp(-0.25*x*x)
    # return x * tf.exp(-0.125 * x * x)
    return x * tf.exp(-0.075*x * x)
    # return -1.25*x*tf.exp(-0.25*x*x)


def sm_mexican(x):
    # return tf.sin(np.pi*x) * x * tf.exp(-0.075*x * x)
    # return tf.sin(np.pi*x) * x * tf.exp(-0.125*x * x)
    return 2.0*tf.sin(np.pi*x) * x * tf.exp(-0.5*x * x)


def singauss(x):
    return tf.exp(-0.25*x*x)*tf.sin(np.pi*x)
    # return tf.sin(2*np.pi*tf.exp(-0.5*x*x))


def powsin_srelu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)*tf.sin(2*np.pi*x)


def sin2_srelu(x):
    return 2.0*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(4*np.pi*x)*tf.sin(2*np.pi*x)


def slrelu(x):
    return tf.nn.leaky_relu(1-x)*tf.nn.leaky_relu(x)


def pow2relu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.nn.relu(x)


def selu(x):
    return tf.nn.elu(1-x)*tf.nn.elu(x)


def wave(x):
    return tf.nn.relu(x) - 2*tf.nn.relu(x-1/4) + \
           2*tf.nn.relu(x-3/4) - tf.nn.relu(x-1)


def phi(x):
    return tf.nn.relu(x) * tf.nn.relu(x)-3*tf.nn.relu(x-1)*tf.nn.relu(x-1) + 3*tf.nn.relu(x-2)*tf.nn.relu(x-2) \
           - tf.nn.relu(x-3)*tf.nn.relu(x-3)*tf.nn.relu(x-3)


#  ------------------------------------------------  初始化kernel --------------------------------------------
# tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，
# 均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，
# 那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
# truncated_normal(
#     shape,
#     mean=0.0,
#     stddev=1.0,
#     dtype=tf.float32,
#     seed=None,
#     name=None)
def truncated_normal_init(h2kernel, w2kernel, in_dim, out_dim, scale_coef=1.0, kernel_name='kernel'):
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    # 尺度因子防止初始化的数值太小或者太大
    V = tf.Variable(
        scale_coef*tf.truncated_normal([h2kernel, w2kernel, in_dim, out_dim], stddev=xavier_stddev),
        dtype=tf.float32, name=kernel_name)
    return V


# tf.random_uniform()
# 默认是在 0 到 1 之间产生随机数，也可以通过 minval 和 maxval 指定上下界
def uniform_init(h2kernel, w2kernel, in_dim, out_dim, kernel_name='kernel'):
    V = tf.Variable(tf.random_uniform([h2kernel, w2kernel, in_dim, out_dim], dtype=tf.float32),
                    dtype=tf.float32, name=kernel_name)
    return V


# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# 从正态分布中输出随机值。
# 参数:
#     shape: 一维的张量，也是输出的张量。
#     mean: 正态分布的均值。
#     stddev: 正态分布的标准差。
#     dtype: 输出的类型。
#     seed: 一个整数，当设置之后，每次生成的随机数都一样。
#     name: 操作的名字。
def normal_init(h2kernel, w2kernel, in_dim, out_dim, scale_coef=1.0, kernel_name='kernel'):
    stddev2normal = np.sqrt(2.0/(in_dim + out_dim))
    # 尺度因子防止初始化的数值太小或者太大
    V = tf.Variable(
        scale_coef*tf.random_normal([h2kernel, w2kernel, in_dim, out_dim], mean=0, stddev=stddev2normal, dtype=tf.float32),
        dtype=tf.float32, name=kernel_name)
    return V


# tf.zeros(
#     shape,
#     dtype=tf.float32,
#     name=None
# )
# shape代表形状，也就是1纬的还是2纬的还是n纬的数组
def zeros_init(h2kernel, w2kernel, in_dim, out_dim, kernel_name='kernel'):
    V = tf.Variable(tf.zeros([h2kernel, w2kernel, in_dim, out_dim], dtype=tf.float32), dtype=tf.float32, name=kernel_name)
    return V


# 生成CNN的kernel
# tf.random_normal(): 用于从服从指定正太分布的数值中取出随机数
# tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
# shape: 输出张量的形状，必选.--- mean: 正态分布的均值，默认为0.----stddev: 正态分布的标准差，默认为1.0
# dtype: 输出的类型，默认为tf.float32 ----seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样---name: 操作的名称
def Initial_Kernel2different_hidden(height2kernel=1, width2kernel=1, in_size=2, out_size=1, hidden_layers=None,
                                    Flag=None):
    n_hiddens = len(hidden_layers)
    Kernels = []  # 权重列表，用于存储隐藏层的权重
    Biases = []  # 偏置列表，用于存储隐藏层的偏置
    # 隐藏层：第一层的权重和偏置，对输入数据做变换
    W = tf.Variable(0.1 * tf.random.normal([height2kernel, width2kernel, in_size, hidden_layers[0]]), dtype='float32',
                    name='kernel_transInput' + str(Flag))
    B = tf.Variable(0.1 * tf.random.uniform([1, 1, 1, hidden_layers[0]]), dtype='float32',
                    name='B_transInput' + str(Flag))
    Kernels.append(W)
    Biases.append(B)
    # 隐藏层：第二至倒数第二层的权重和偏置
    for i_layer in range(n_hiddens - 1):
        W = tf.Variable(
            0.1 * tf.random.normal([height2kernel, width2kernel, hidden_layers[i_layer], hidden_layers[i_layer+1]]),
            dtype='float32', name='kernel_hidden' + str(i_layer + 1) + str(Flag))
        B = tf.Variable(0.1 * tf.random.uniform([1, 1, 1, hidden_layers[i_layer+1]]), dtype='float32',
                        name='B_hidden' + str(i_layer + 1) + str(Flag))
        Kernels.append(W)
        Biases.append(B)

    # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
    W = tf.Variable(0.1 * tf.random.normal([height2kernel, width2kernel, hidden_layers[-1], out_size]), dtype='float32',
                    name='W_outTrans' + str(Flag))
    B = tf.Variable(0.1 * tf.random.uniform([1, 1, 1, out_size]), dtype='float32',
                    name='B_outTrans' + str(Flag))
    Kernels.append(W)
    Biases.append(B)

    return Kernels, Biases


def initialize_Kernel_xavier(height2kernel=1, width2kernel=1, in_size=2, out_size=1, hidden_layers=None, Flag=None):
    with tf.variable_scope('kernel_scope', reuse=tf.AUTO_REUSE):
        scale = 5.0
        n_hiddens = len(hidden_layers)
        Kernels = []                  # kernel列表，用于存储隐藏层的权重
        Biases = []

        # 隐藏层：第一层的kernel和偏置，对输入数据做变换
        W = truncated_normal_init(height2kernel, width2kernel, in_size, hidden_layers[0],
                                  scale_coef=scale, kernel_name='kernel-transInput' + str(Flag))
        B = uniform_init(height2kernel, width2kernel, 1, hidden_layers[0], kernel_name='B-transInput' + str(Flag))
        Kernels.append(W)
        Biases.append(B)
        for i_layer in range(0, n_hiddens - 1):
            W = truncated_normal_init(height2kernel, width2kernel, hidden_layers[i_layer], hidden_layers[i_layer + 1],
                                      scale_coef=scale, kernel_name='kernel-hidden' + str(i_layer + 1) + str(Flag))
            B = uniform_init(height2kernel, width2kernel, 1, hidden_layers[i_layer + 1],
                             kernel_name='B-hidden' + str(i_layer + 1) + str(Flag))
            Kernels.append(W)
            Biases.append(B)

        # 输出层：最后一层的kernel和偏置。将最后的结果变换到输出维度
        W = truncated_normal_init(height2kernel, width2kernel, hidden_layers[-1], out_size, scale_coef=scale,
                                  kernel_name='kernel-outTrans' + str(Flag))
        B = uniform_init(height2kernel, width2kernel, 1, out_size, kernel_name='B-outTrans' + str(Flag))
        Kernels.append(W)
        Biases.append(B)
        return Kernels, Biases


def initialize_Kernel_random_normal(height2kernel=1, width2kernel=1, in_size=2, out_size=1, hidden_layers=None,
                                    Flag=None, varcoe=0.5):
    with tf.variable_scope('kernel_scope', reuse=tf.AUTO_REUSE):
        n_hiddens = len(hidden_layers)
        Kernels = []  # 权重列表，用于存储隐藏层的权重
        Biases = []   # 偏置列表，用于存储隐藏层的偏置
        # 隐藏层：第一层的权重和偏置，对输入数据做变换
        stddev_WB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.get_variable(name='kernel-transInput' + str(Flag),
                            shape=(height2kernel, width2kernel, in_size, hidden_layers[0]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        B = tf.get_variable(name='B-transInput' + str(Flag),
                            shape=(height2kernel, width2kernel, 1, hidden_layers[0]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        Kernels.append(W)
        Biases.append(B)
        for i_layer in range(0, n_hiddens - 1):
            stddev_WB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
            W = tf.get_variable(name='kernel' + str(i_layer + 1) + str(Flag),
                                shape=(height2kernel, width2kernel, hidden_layers[i_layer], hidden_layers[i_layer + 1]),
                                initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                dtype=tf.float32)
            B = tf.get_variable(name='B' + str(i_layer + 1) + str(Flag),
                                shape=(height2kernel, width2kernel, 1, hidden_layers[i_layer + 1]),
                                initializer=tf.random_normal_initializer(stddev=stddev_WB),
                                dtype=tf.float32)
            Kernels.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        stddev_WB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.get_variable(name='kernel-outTrans' + str(Flag),
                            shape=(height2kernel, width2kernel, hidden_layers[-1], out_size),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        B = tf.get_variable(name='B-outTrans' + str(Flag),
                            shape=(height2kernel, width2kernel, 1, out_size),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)

        Kernels.append(W)
        Biases.append(B)
        return Kernels, Biases


def initialize_Kernel_random_normal2(height2kernel=1, width2kernel=1, in_size=2, out_size=1, hidden_layers=None,
                                     Flag=None, varcoe=0.5):
    with tf.variable_scope('Kernel_scope', reuse=tf.AUTO_REUSE):
        n_hiddens = len(hidden_layers)
        kernels = []  # Kernel列表，用于存储隐藏层的权重
        Biases = []  # 偏置列表，用于存储隐藏层的偏置
        # 隐藏层：第一层的kernel，对输入数据做变换
        stddev_KB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.get_variable(name='Kernel-transInput' + str(Flag),
                            shape=(height2kernel, width2kernel, in_size, hidden_layers[0]),
                            initializer=tf.random_normal_initializer(stddev=stddev_KB),
                            dtype=tf.float32)
        B = tf.get_variable(name='B-transInput' + str(Flag),
                            shape=(1, 1, 1, hidden_layers[0]),
                            initializer=tf.random_normal_initializer(stddev=stddev_KB),
                            dtype=tf.float32)
        kernels.append(W)
        Biases.append(B)
        for i_layer in range(0, n_hiddens - 1):
            stddev_KB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
            W = tf.get_variable(name='Kernel' + str(i_layer + 1) + str(Flag),
                                shape=(height2kernel, width2kernel, hidden_layers[i_layer], hidden_layers[i_layer + 1]),
                                initializer=tf.random_normal_initializer(stddev=stddev_KB),
                                dtype=tf.float32)
            B = tf.get_variable(name='B-hidden' + str(i_layer + 1) + str(Flag),
                                shape=(1, 1, 1, hidden_layers[i_layer + 1]),
                                initializer=tf.random_normal_initializer(stddev=stddev_KB),
                                dtype=tf.float32)
            kernels.append(W)
            Biases.append(B)
        # 输出层：最后一层B的kernel。将最后的结果变换到输出维度
        stddev_KB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.get_variable(name='Kernel-outTrans' + str(Flag),
                            shape=(height2kernel, width2kernel, hidden_layers[-1], out_size),
                            initializer=tf.random_normal_initializer(stddev=stddev_KB),
                            dtype=tf.float32)
        B = tf.get_variable(name='B-outTrans' + str(Flag),
                            shape=(1, 1, 1, out_size),
                            initializer=tf.random_normal_initializer(stddev=stddev_KB),
                            dtype=tf.float32)

        kernels.append(W)
        Biases.append(B)
        return kernels, Biases


def weight_variable(shape, name2weight=None):
    if len(shape) == 4:
        xavier_stddev = np.sqrt(2 / (shape[2] + shape[3]))
    else:
        xavier_stddev = 0.1
    initial_V = tf.Variable(0.25*tf.truncated_normal(shape, stddev=xavier_stddev), dtype=tf.float32, name=name2weight)
    return initial_V


def construct_weights2kernel(height2kernel=1, width2kernel=1, hidden_layers=None,  in_channel=1, out_channel=2):
    layers = int(len(hidden_layers))

    kernels = []

    kernel = weight_variable([height2kernel, width2kernel, int(in_channel), hidden_layers[0]], name2weight='W_in')
    kernels.append(kernel)
    for i_layer in range(layers - 1):
        kernel = weight_variable([height2kernel, width2kernel, hidden_layers[i_layer], hidden_layers[i_layer + 1]], name2weight='W'+str(i_layer+1))
        kernels.append(kernel)

    kernel = weight_variable([height2kernel, width2kernel, hidden_layers[-1], int(out_channel)], name2weight='W_out')
    kernels.append(kernel)
    return kernels


# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
# 除去name参数用以指定该操作的name，与方法有关的一共五个参数：
#
# 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
# 具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
#
# 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]
# 这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，
# 第三维in_channels，就是参数input的第四维
#
# 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
#
# 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
#
# 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
#
# 结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。

def my_conv2d(v_input, kernel=None, bias2conv=None, activate_name=None, if_scale=False, freqs2scale=None):
    """Construct new-point feature for each point
        Args:
        v_input: (num_points, num_dims) or (1, 1, num_points, num_dims)
        kernel: (num_dims, new_num_dims) or (1, *, num_points, new_num_dims)
        bias2conv: (1, 1, 1, new_num_dims)
        activate_name: the name of activation function
        if_scale: bool -- whether need scale transform for input
        freqs2scale: the array of

        Returns:
        v_output: (1,1, num_points, new_num_dims)
    """
    if activate_name == 'relu':
        DNN_activation = tf.nn.relu
    elif activate_name == 'leaky_relu':
        DNN_activation = tf.nn.leaky_relu(0.2)
    elif activate_name == 'srelu':
        DNN_activation = srelu
    elif activate_name == 's2relu':
        DNN_activation = s2relu
    elif activate_name == 'sin2_srelu':
        DNN_activation = sin2_srelu
    elif activate_name == 'powsin_srelu':
        DNN_activation = powsin_srelu
    elif activate_name == 'slrelu':
        DNN_activation = slrelu
    elif activate_name == 'elu':
        DNN_activation = tf.nn.elu
    elif activate_name == 'selu':
        DNN_activation = selu
    elif activate_name == 'sin':
        DNN_activation = mysin
    elif activate_name == 'tanh':
        DNN_activation = tf.nn.tanh
    elif activate_name == 'sintanh':
        DNN_activation = stanh
    elif activate_name == 'gauss':
        DNN_activation = gauss
    elif activate_name == 'singauss':
        DNN_activation = singauss
    elif activate_name == 'mexican':
        DNN_activation = mexican
    elif activate_name == 'modify_mexican':
        DNN_activation = modify_mexican
    elif activate_name == 'sin_modify_mexican':
        DNN_activation = sm_mexican
    elif activate_name == 'phi':
        DNN_activation = phi

    shape2v = v_input.get_shape()
    if len(shape2v) == 2:
        v_input = tf.expand_dims(v_input, axis=0)
        v_input = tf.expand_dims(v_input, axis=0)
    elif len(shape2v) == 3:
        v_input = tf.expand_dims(v_input, axis=0)

    shape2kernel = kernel.get_shape()
    assert(len(shape2kernel)) == 4
    outputs = tf.nn.conv2d(v_input, kernel, strides=[1, 1, 1, 1], padding="SAME")

    shape2out = outputs.get_shape()
    # shape2out = out.get_shape().as_list()
    assert(len(shape2out)) == 4
    shape2bias = bias2conv.get_shape()
    assert (len(shape2bias)) == 4
    assert (shape2out[-1] == shape2bias[-1])
    if if_scale:
        # print('shape2out[-1]', shape2out[-1])
        Unit_num = int(shape2out[-1] // len(freqs2scale))

        # np.repeat(a, repeats, axis=None)
        # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
        # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
        mixcoe = np.repeat(freqs2scale, Unit_num)

        # 这个的作用是什么？
        mixcoe = np.concatenate((mixcoe, np.ones([shape2out[-1] - Unit_num * len(freqs2scale)]) * freqs2scale[-1]))

        mixcoe = mixcoe.astype(np.float32)
        mixcoe = np.expand_dims(mixcoe, axis=0)
        mixcoe = np.expand_dims(mixcoe, axis=0)
        mixcoe = np.expand_dims(mixcoe, axis=0)
        outputs = tf.multiply(outputs, mixcoe)

    out_result = DNN_activation(tf.add(outputs, bias2conv))

    # out_result = tf.reduce_mean(outputs, axis=0, keep_dims=True)
    return out_result


def _variable_on_cpu(name, shape, initializer, use_fp16=False, trainable=True):
    """Helper to create a Variable stored on CPU memory.
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

    Returns:
    Variable Tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           freqs=None,
           scale_trans=False,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
    """ 2D convolution with non-linear operation.

      Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

      Returns:
        Variable tensor
      """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)

        if scale_trans:
            shape2out = outputs.get_shape()
            Unit_num = int(shape2out[0] / len(freqs))

            # np.repeat(a, repeats, axis=None)
            # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
            # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
            mixcoe = np.repeat(freqs, Unit_num)

            # 这个的作用是什么？
            mixcoe = np.concatenate((mixcoe, np.ones([shape2out[0] - Unit_num * len(freqs)]) * freqs[-1]))

            mixcoe = mixcoe.astype(np.float32)
            mixcoe = np.expand_dims(mixcoe, axis=0)
            mixcoe = np.expand_dims(mixcoe, axis=0)
            mixcoe = np.expand_dims(mixcoe, axis=0)
            outputs = tf.multiply(outputs, mixcoe)

        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn', is_dist=is_dist)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d_nobias(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
    """ 2D convolution with non-linear operation.

      Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

      Returns:
        Variable tensor
      """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn', is_dist=is_dist)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_dist_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ The batch normalization for distributed training.
    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = _variable_on_cpu('beta', [num_channels], initializer=tf.zeros_initializer())
        gamma = _variable_on_cpu('gamma', [num_channels], initializer=tf.ones_initializer())

        pop_mean = _variable_on_cpu('pop_mean', [num_channels], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = _variable_on_cpu('pop_var', [num_channels], initializer=tf.ones_initializer(), trainable=False)

        def train_bn_op():
            batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
            decay = bn_decay if bn_decay is not None else 0.9
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, 1e-3)

        def test_bn_op():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, 1e-3)

        normed = tf.cond(is_training,
                         train_bn_op,
                         test_bn_op)
        return normed


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, is_dist=False):
    """ Batch normalization on 2D convolutional maps.

    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
        is_dist:     true indicating distributed training scheme
    Return:
        normed:      batch-normalized maps
    """
    if is_dist:
        return batch_norm_dist_template(inputs, is_training, scope, [0, 1, 2], bn_decay)
    else:
        return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay)


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.

    Args:
      point_cloud: tensor (num_points, dims2point)

    Returns:
      pairwise distance: (num_points, num_points)
    """
    point_cloud_shape = point_cloud.get_shape().as_list()
    assert(len(point_cloud_shape)) == 2

    point_cloud_transpose = tf.transpose(point_cloud, perm=[1, 0])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square_transpose = tf.transpose(point_cloud_square, perm=[1, 0])
    return point_cloud_square + point_cloud_inner + point_cloud_square_transpose


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
      Args:
        pairwise distance: (batch_size, num_points, num_points)
        k: int

      Returns:
        nearest neighbors: (batch_size, num_points, k)
      """
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k)  # 这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
    return nn_idx


def get_neighbors(point_cloud, nn_idx, k=20):
    """Construct neighbors feature for each point
      Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k)
        k: int

      Returns:
        neighbors features: (batch_size, num_points, k, num_dims)
      """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    og_num_dims = point_cloud.get_shape().as_list()[-1]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)
    if og_num_dims == 1:
        point_cloud = tf.expand_dims(point_cloud, -1)

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)

    return point_cloud_neighbors


# 利用PointNet设计的卷积函数进行卷积运算，卷积核和偏置是在卷积函数中生成的
def PDE_SingleGNN_conv2d(point_cloud, nn_idx=None, k_neighbors=20, dim2out=10, act_name2stretch=None,
                         act_func2neigh_atten=tf.nn.relu, is_training=None, bn_decay=False, scope2atten='122132',
                         trans_dim=False):

    """"Construct edge feature for each point
    Args:
    point_cloud: float array -- (num_points, num_dims)
    nn_idx: int array --(num_points, k)
    k_neighbors: int
    dim2out: the out-dimension for stretching variable
    act_func2stretch: string -- the activation function for  stretching the dimension of input-variable
    act_func2neigh_atten: the activation function of obtaining the coefficient for different neighbor point
    is_training: bool
    bn_decay: bool
    scope2atten: string -- the namespace of both stretch-variable and aggregate-coefficient

    Returns:
    new point_cloud: (num_points, out_dim)
    """
    if act_name2stretch == 'relu':
        act_func2stretch = tf.nn.relu
    elif act_name2stretch == 'leaky_relu':
        act_func2stretch = tf.nn.leaky_relu(0.2)
    elif act_name2stretch == 'srelu':
        act_func2stretch = srelu
    elif act_name2stretch == 's2relu':
        act_func2stretch = s2relu
    elif act_name2stretch == 'sin2_srelu':
        act_func2stretch = sin2_srelu
    elif act_name2stretch == 'powsin_srelu':
        act_func2stretch = powsin_srelu
    elif act_name2stretch == 'slrelu':
        act_func2stretch = slrelu
    elif act_name2stretch == 'elu':
        act_func2stretch = tf.nn.elu
    elif act_name2stretch == 'selu':
        act_func2stretch = selu
    elif act_name2stretch == 'sin':
        act_func2stretch = mysin
    elif act_name2stretch == 'tanh':
        act_func2stretch = tf.nn.tanh
    elif act_name2stretch == 'sintanh':
        act_func2stretch = stanh
    elif act_name2stretch == 'gauss':
        act_func2stretch = gauss
    elif act_name2stretch == 'singauss':
        act_func2stretch = singauss
    elif act_name2stretch == 'mexican':
        act_func2stretch = mexican
    elif act_name2stretch == 'modify_mexican':
        act_func2stretch = modify_mexican
    elif act_name2stretch == 'sin_modify_mexican':
        act_func2stretch = sm_mexican
    elif act_name2stretch == 'phi':
        act_func2stretch = phi

    point_cloud_shape = point_cloud.get_shape()
    assert (len(point_cloud_shape)) == 2
    num_points = point_cloud_shape[0].value
    num_dims = point_cloud_shape[1].value
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        # 对原始点集升维
        if dim2out > num_dims:
            print('raising the dimension of variable')
            if trans_dim:
                expand_point_cloud = tf.expand_dims(tf.expand_dims(point_cloud, axis=0), axis=-2)
                # 每次调用这个函数，scope2atten 不能相同
                new_points_cloud = conv2d(expand_point_cloud, dim2out, [1, 1], padding='VALID', stride=[1, 1],
                                          activation_fn=act_func2stretch, bn=False, is_training=is_training,
                                          scope='new_points_cloud' + scope2atten, bn_decay=bn_decay)
                squeeze_new_points = tf.squeeze(new_points_cloud)
            else:
                squeeze_new_points = point_cloud
        elif dim2out <= num_dims:  # 对原始点集降维或者维持维度不变
            print('reducing or keeping the dimension of variable')
            if trans_dim:
                expand_point_cloud = tf.expand_dims(tf.expand_dims(point_cloud, axis=0), axis=-2)
                # 每次调用这个函数，scope2atten 不能相同
                new_points_cloud = conv2d(expand_point_cloud, dim2out, [1, 1], padding='VALID', stride=[1, 1],
                                          activation_fn=act_func2stretch, bn=False, is_training=is_training,
                                          scope='new_points_cloud' + scope2atten, bn_decay=bn_decay)
                squeeze_new_points = tf.squeeze(new_points_cloud)
            else:
                squeeze_new_points = point_cloud

        # 选出每个点的邻居，然后的到每个点和各自邻居点的边
        select_idx = nn_idx + num_points                               # 邻居点的indexes
        point_neighbors = tf.gather(squeeze_new_points, select_idx)    # 邻居点的'坐标'

        point_central = tf.expand_dims(squeeze_new_points, axis=-2)    # 每个点作为中心点，在倒数第二个位置扩维
        centroids_tilde = tf.tile(point_central, [1, k_neighbors, 1])  # 中心点复制k份
        edges_feature = centroids_tilde - point_neighbors              # 中心点和各自的邻居点形成的边

        # 根据边的长度计算每个邻居点对应的权重系数，然后根据系数计算邻居点对中心点的贡献
        expand_edges = tf.expand_dims(edges_feature, axis=0)  # conv2d 函数接收的输入是一个 4d tensor，先将edges_feature扩维
        # 每次调用这个函数，scope2atten 不能相同
        atten_ceof2neighbors = conv2d(expand_edges, 1, [1, 1], padding='VALID', stride=[1, 1],
                                      activation_fn=act_func2neigh_atten, bn=False, is_training=is_training,
                                      scope='neighbors_attention' + scope2atten,
                                      bn_decay=bn_decay)
        # 归一化得到注意力系数
        atten_ceof2neighbors = tf.nn.softmax(atten_ceof2neighbors)
        neighbors_coef = tf.transpose(atten_ceof2neighbors, perm=[0, 1, 3, 2])

        expand_neighbors = tf.expand_dims(point_neighbors, axis=0)
        atten_neighbors = tf.matmul(neighbors_coef, expand_neighbors)   # 利用注意力系数将各个邻居聚合起来

        squeeze2atten_neighbors = tf.squeeze(atten_neighbors)           # 去除数值为1的维度

        # 中心点的特征和邻居点的特征相加，得到新的特征
        new_point_cloud = tf.add(squeeze_new_points, squeeze2atten_neighbors)

    return new_point_cloud


# 利用PointNet设计的卷积函数进行卷积运算，卷积核和偏置是在卷积函数中生成的
def PDE_HierarchicGNN_conv2d(point_cloud, nn_idx=None, k_neighbors=20, dim2out=10, act_name2stretch=None, hiddens=None,
                            act_func2neigh_atten=tf.nn.relu, is_training=None, bn_decay=False, scope2atten='122132',
                            trans_dim=False):

    """"Construct edge feature for each point
    Args:
    point_cloud: float array -- (num_points, num_dims)
    nn_idx: int array --(num_points, k)
    k_neighbors: int
    dim2out: the out-dimension for stretching variable
    act_func2stretch: string -- the activation function for  stretching the dimension of input-variable
    act_func2neigh_atten: the activation function of obtaining the coefficient for different neighbor point
    is_training: bool
    bn_decay: bool
    scope2atten: string -- the namespace of both stretch-variable and aggregate-coefficient

    Returns:
    new point_cloud: (num_points, out_dim)
    """
    if act_name2stretch == 'relu':
        act_func2stretch = tf.nn.relu
    elif act_name2stretch == 'leaky_relu':
        act_func2stretch = tf.nn.leaky_relu(0.2)
    elif act_name2stretch == 'srelu':
        act_func2stretch = srelu
    elif act_name2stretch == 's2relu':
        act_func2stretch = s2relu
    elif act_name2stretch == 'sin2_srelu':
        act_func2stretch = sin2_srelu
    elif act_name2stretch == 'powsin_srelu':
        act_func2stretch = powsin_srelu
    elif act_name2stretch == 'slrelu':
        act_func2stretch = slrelu
    elif act_name2stretch == 'elu':
        act_func2stretch = tf.nn.elu
    elif act_name2stretch == 'selu':
        act_func2stretch = selu
    elif act_name2stretch == 'sin':
        act_func2stretch = mysin
    elif act_name2stretch == 'tanh':
        act_func2stretch = tf.nn.tanh
    elif act_name2stretch == 'sintanh':
        act_func2stretch = stanh
    elif act_name2stretch == 'gauss':
        act_func2stretch = gauss
    elif act_name2stretch == 'singauss':
        act_func2stretch = singauss
    elif act_name2stretch == 'mexican':
        act_func2stretch = mexican
    elif act_name2stretch == 'modify_mexican':
        act_func2stretch = modify_mexican
    elif act_name2stretch == 'sin_modify_mexican':
        act_func2stretch = sm_mexican
    elif act_name2stretch == 'phi':
        act_func2stretch = phi

    point_cloud_shape = point_cloud.get_shape()
    assert (len(point_cloud_shape)) == 2
    num_points = point_cloud_shape[0].value
    num_dims = point_cloud_shape[1].value
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        # 对原始点集升维
        if dim2out > num_dims:
            print('raising the dimension of variable')
            if trans_dim:
                expand_point_cloud = tf.expand_dims(tf.expand_dims(point_cloud, axis=0), axis=-2)
                # 每次调用这个函数，scope2atten 不能相同
                new_points_cloud = conv2d(expand_point_cloud, dim2out, [1, 1], padding='VALID', stride=[1, 1],
                                          activation_fn=act_func2stretch, bn=False, is_training=is_training,
                                          scope='new_points_cloud' + scope2atten, bn_decay=bn_decay)
                squeeze_new_points = tf.squeeze(new_points_cloud)
            else:
                squeeze_new_points = point_cloud
        elif dim2out <= num_dims:  # 对原始点集降维或者维持维度不变
            print('reducing or keeping the dimension of variable')
            if trans_dim:
                expand_point_cloud = tf.expand_dims(tf.expand_dims(point_cloud, axis=0), axis=-2)
                # 每次调用这个函数，scope2atten 不能相同
                new_points_cloud = conv2d(expand_point_cloud, dim2out, [1, 1], padding='VALID', stride=[1, 1],
                                          activation_fn=act_func2stretch, bn=False, is_training=is_training,
                                          scope='new_points_cloud' + scope2atten, bn_decay=bn_decay)
                squeeze_new_points = tf.squeeze(new_points_cloud)
            else:
                squeeze_new_points = point_cloud

        # 选出每个点的邻居，然后的到每个点和各自邻居点的边
        select_idx = nn_idx + num_points                               # 邻居点的indexes
        point_neighbors = tf.gather(squeeze_new_points, select_idx)    # 邻居点的'坐标'

        point_central = tf.expand_dims(squeeze_new_points, axis=-2)    # 每个点作为中心点，在倒数第二个位置扩维
        centroids_tilde = tf.tile(point_central, [1, k_neighbors, 1])  # 中心点复制k份
        edges_feature = centroids_tilde - point_neighbors              # 中心点和各自的邻居点形成的边

        # 根据边的长度计算每个邻居点对应的权重系数，然后根据系数计算邻居点对中心点的贡献
        expand_edges = tf.expand_dims(edges_feature, axis=0)  # conv2d 函数接收的输入是一个 4d tensor，先将edges_feature扩维
        # 每次调用这个函数，scope2atten 不能相同
        atten_ceof2neighbors = conv2d(expand_edges, 1, [1, 1], padding='VALID', stride=[1, 1],
                                      activation_fn=act_func2neigh_atten, bn=False, is_training=is_training,
                                      scope='neighbors_attention' + scope2atten,
                                      bn_decay=bn_decay)
        # 归一化得到注意力系数
        atten_ceof2neighbors = tf.nn.softmax(atten_ceof2neighbors)
        neighbors_coef = tf.transpose(atten_ceof2neighbors, perm=[0, 1, 3, 2])

        expand_neighbors = tf.expand_dims(point_neighbors, axis=0)
        atten_neighbors = tf.matmul(neighbors_coef, expand_neighbors)   # 利用注意力系数将各个邻居聚合起来

        squeeze2atten_neighbors = tf.squeeze(atten_neighbors)           # 去除数值为1的维度

        # 中心点的特征和邻居点的特征相加，得到新的特征
        new_point_cloud = tf.add(squeeze_new_points, squeeze2atten_neighbors)

    return new_point_cloud


# 利用输入的卷积核和偏置进行卷积运算，卷积函数使用本人自定义的
def PDE_SingleGNN_myConv(point_cloud, weight=None, bias=None, nn_idx=None, k_neighbors=20, act_name2stretch=None,
                         trans_dim=False, act_func2neigh_atten=tf.nn.relu, is_training=None, bn_decay=False,
                         scope2atten='neigh_atten'):

    """Construct edge feature for each point
    Args:
    point_cloud: float array -- (num_points, num_dims)
    nn_idx: int array --(num_points, k)
    k_neighbors: int
    weight: (1, 1, num_dims, out_dim)
    bias:  (1, 1, 1, out_dim)
    act_name2stretch: string -- the name of activation function for  stretching the dimension of input-variable
    act2neigh_atten: the activation function of obtaining the coefficient for different neighbor point
    is_training: bool
    bn_decay: bool
    scope2atten: string -- the namespace of attention coefficient for aggregating neighbors

    Returns:
    new point_cloud: (num_points, out_dim)
    """
    point_cloud_shape = point_cloud.get_shape()
    assert (len(point_cloud_shape)) == 2
    num_points = point_cloud_shape[0].value
    num_dims = point_cloud_shape[1].value

    weight_shape = weight.get_shape()
    if len(weight_shape) == 2:
        weight = tf.expand_dims(weight, axis=0)
        weight = tf.expand_dims(weight, axis=0)
    elif len(weight_shape) == 3:
        weight = tf.expand_dims(weight, axis=0)
    else:
        assert(len(weight_shape)) == 4
    bias_shape = bias.get_shape()
    if len(bias_shape) == 2:
        bias = tf.expand_dims(bias, axis=0)
        bias = tf.expand_dims(bias, axis=0)
    elif len(bias_shape) == 3:
        bias = tf.expand_dims(bias, axis=0)
    else:
        assert (len(bias_shape)) == 4

    out_dim = weight_shape[-1]
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        # 对原始点集升维
        if out_dim > num_dims:
            print('raising the dimension of variable')
            if trans_dim:
                expand_point_cloud = tf.expand_dims(tf.expand_dims(point_cloud, axis=0), axis=-2)
                new_points_cloud = my_conv2d(expand_point_cloud, kernel=weight, bias2conv=bias,
                                             activate_name=act_name2stretch)
                squeeze_new_points = tf.squeeze(new_points_cloud)
            else:
                squeeze_new_points = point_cloud
        elif out_dim <= num_dims:      # 对原始点集降维或者维持维度不变
            print('reducing or keeping the dimension of variable')
            if trans_dim:
                expand_point_cloud = tf.expand_dims(tf.expand_dims(point_cloud, axis=0), axis=-2)
                new_points_cloud = my_conv2d(expand_point_cloud, kernel=weight, bias2conv=bias,
                                             activate_name=act_name2stretch)
                squeeze_new_points = tf.squeeze(new_points_cloud)
            else:
                squeeze_new_points = point_cloud

        # 选出每个点的邻居，然后的到每个点和各自邻居点的边
        select_idx = nn_idx + num_points                               # 邻居点的indexes
        point_neighbors = tf.gather(squeeze_new_points, select_idx)    # 邻居点的'坐标'

        point_central = tf.expand_dims(squeeze_new_points, axis=-2)    # 每个点作为中心点，在倒数第二个位置扩维
        centroids_tilde = tf.tile(point_central, [1, k_neighbors, 1])  # 中心点复制k份
        edges_feature = centroids_tilde - point_neighbors              # 中心点和各自的邻居点形成的边

        # 根据边的长度计算每个邻居点对应的权重系数，然后根据系数计算邻居点对中心点的贡献
        expand_edges = tf.expand_dims(edges_feature, axis=0)   # conv2d 函数接收的输入是一个 4d tensor，先将edges_feature扩维
        # 每次调用这个函数，scope2atten 不能相同
        atten_ceof2neighbors = conv2d(expand_edges, 1, [1, 1], padding='VALID', stride=[1, 1],
                                      activation_fn=act_func2neigh_atten, bn=False, is_training=is_training,
                                      scope='neighbors_attention' + scope2atten, bn_decay=bn_decay)

        atten_ceof2neighbors = tf.nn.softmax(atten_ceof2neighbors)
        neighbors_coef = tf.transpose(atten_ceof2neighbors, perm=[0, 1, 3, 2])

        expand_neighbors = tf.expand_dims(point_neighbors, axis=0)
        atten_neighbors = tf.matmul(neighbors_coef, expand_neighbors)   # 利用注意力系数将各个邻居聚合起来

        squeeze2atten_neighbors = tf.squeeze(atten_neighbors)           # 去除数值为1的维度

        # 中心点的特征和邻居点的特征相加，得到新的特征
        new_point_cloud = tf.add(squeeze_new_points, squeeze2atten_neighbors)

    return new_point_cloud


# 利用输入的卷积核和偏置进行卷积运算，卷积函数使用本人自定义的
def PDE_HierarchicGNN_myConv(point_set, Weight_list=None, Bias_list=None, nn_idx=None, k_neighbors=20,
                             act_name2stretch=None, trans_dim=False, scale_trans=False, freqs=None,
                             scope2atten='neighbor', act_func2neigh_atten=tf.nn.relu, is_training=None, bn_decay=False):

    """Construct edge feature for each point
    Args:
    input_points: float array -- (num_points, num_dims)
    nn_idx: int array --(num_points, k)
    k_neighbors: int
    weight: (1, 1, num_dims, out_dim)
    bias:  (1, 1, 1, out_dim)
    act_name2stretch: string -- the name of activation function for  stretching the dimension of input-variable
    act2neigh_atten: the activation function of obtaining the coefficient for different neighbor point
    scale_trans: bool
    freqs: array
    is_training: bool
    bn_decay: bool
    scope2atten: string -- the namespace of attention coefficient for aggregating neighbors

    Returns:
    new point_cloud: (num_points, out_dim)
    """
    point_set_shape = point_set.get_shape()
    assert (len(point_set_shape)) == 2
    num_points = point_set_shape[0].value
    num_dims = point_set_shape[1].value

    out_set = point_set
    len2weigth_list = len(Weight_list)
    for i_layer in range(len2weigth_list):
        weight = Weight_list[i_layer]
        bias = Bias_list[i_layer]
        weight_shape = weight.get_shape()
        if len(weight_shape) == 2:
            weight = tf.expand_dims(weight, axis=0)
            weight = tf.expand_dims(weight, axis=0)
        elif len(weight_shape) == 3:
            weight = tf.expand_dims(weight, axis=0)
        else:
            assert(len(weight_shape)) == 4

        bias_shape = bias.get_shape()
        if len(bias_shape) == 2:
            bias = tf.expand_dims(bias, axis=0)
            bias = tf.expand_dims(bias, axis=0)
        elif len(bias_shape) == 3:
            bias = tf.expand_dims(bias, axis=0)
        else:
            assert (len(bias_shape)) == 4

        out_dim = weight_shape[-1]
        with tf.variable_scope('attention2' + scope2atten, reuse=tf.AUTO_REUSE):
            # 对原始点集升维
            if out_dim > num_dims:
                print('raising the dimension of variable')
                if trans_dim:
                    new_points_cloud = my_conv2d(out_set, kernel=weight, bias2conv=bias,
                                                 activate_name=act_name2stretch, if_scale=scale_trans,
                                                 freqs2scale=freqs)
                    squeeze_new_points = tf.squeeze(tf.squeeze(new_points_cloud, axis=0), axis=0)
                else:
                    squeeze_new_points = out_set
            elif out_dim <= num_dims:                                  # 对原始点集降维或者维持维度不变
                print('reducing or keeping the dimension of variable')
                if trans_dim:
                    new_points_cloud = my_conv2d(out_set, kernel=weight, bias2conv=bias,
                                                 activate_name=act_name2stretch, if_scale=scale_trans,
                                                 freqs2scale=freqs)
                    squeeze_new_points = tf.squeeze(tf.squeeze(new_points_cloud, axis=0), axis=0)
                else:
                    squeeze_new_points = out_set

            assert len(squeeze_new_points.get_shape()) == 2

            # 选出每个点的邻居，然后的到每个点和各自邻居点的边
            select_idx = nn_idx + num_points                               # 邻居点的indexes
            point_neighbors = tf.gather(squeeze_new_points, select_idx)    # 邻居点的'坐标'

            point_central = tf.expand_dims(squeeze_new_points, axis=-2)    # 每个点作为中心点，在倒数第二个位置扩维
            centroids_tilde = tf.tile(point_central, [1, k_neighbors, 1])  # 中心点复制k份
            edges_feature = centroids_tilde - point_neighbors              # 中心点和各自的邻居点形成的边

            # 根据边的长度计算每个邻居点对应的权重系数，然后根据系数计算邻居点对中心点的贡献
            expand_edges = tf.expand_dims(edges_feature, axis=0)   # conv2d 函数接收的输入是一个 4d tensor，先将edges_feature扩维
            # 每次调用这个函数，scope 不能相同
            atten_ceof2neighbors = conv2d(expand_edges, 1, [1, 1], padding='VALID', stride=[1, 1],
                                          activation_fn=act_func2neigh_atten, bn=False, is_training=is_training,
                                          scope='atten2' + scope2atten + str(i_layer), bn_decay=bn_decay)

            atten_ceof2neighbors = tf.nn.softmax(atten_ceof2neighbors)
            neighbors_coef = tf.transpose(atten_ceof2neighbors, perm=[0, 1, 3, 2])

            expand_neighbors = tf.expand_dims(point_neighbors, axis=0)
            atten_neighbors = tf.matmul(neighbors_coef, expand_neighbors)   # 利用注意力系数将各个邻居聚合起来

            squeeze2atten_neighbors = tf.squeeze(tf.squeeze(atten_neighbors, axis=0), axis=1)           # 去除数值为1的维度

            # 中心点的特征和邻居点的特征相加，得到新的特征
            new_point_cloud = tf.add(squeeze_new_points, squeeze2atten_neighbors)

        scale_trans = False
        out_set = new_point_cloud

    return out_set


def test_CNN_model(v_input, kernels=None, act_function=tf.nn.relu):
    # # 后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1。如果是RGB图像，那么channel就是3.
    # out 的第一维是Batch_size大小
    dim2v = v_input.get_shape()
    assert (len(dim2v)) == 2
    if len(dim2v) == 2:
        out = tf.expand_dims(v_input, axis=0)
        out = tf.expand_dims(out, axis=0)

    kernel = kernels[0]
    out = tf.nn.conv2d(out, kernel, strides=[1, 1, 1, 1], padding="SAME")
    out = act_function(out)
    out = tf.reduce_mean(out, axis=0, keep_dims=True)

    len_weigths = len(kernels)
    for i_cnn in range(len_weigths - 2):
        kernel = kernels[i_cnn + 1]
        out = tf.nn.conv2d(out, kernel, strides=[1, 1, 1, 1], padding="SAME")
        out = act_function(out)

    kernel_out = kernels[-1]
    out_result = tf.nn.conv2d(out, kernel_out, strides=[1, 1, 1, 1], padding="SAME")
    return out_result


def test_fun1():
    hidden_layer = (2, 3, 4, 5, 6)
    batchsize_it = 10
    dim_size = 2
    input_dim = 2
    out_dim = 1
    activate_func = tf.nn.relu
    weights2kernel = construct_weights2kernel(
        height2kernel=2, width2kernel=3, hidden_layers=hidden_layer, in_channel=input_dim, out_channel=out_dim)
    with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
        XY_mesh = tf.placeholder(tf.float32, name='X_it', shape=[batchsize_it, dim_size])
        # 在变量的内部区域训练
        U_hat = test_CNN_model(XY_mesh, kernels=weights2kernel, act_function=activate_func)
        U_hat = tf.reduce_mean(U_hat, axis=-1)
        U_hat = tf.squeeze(U_hat)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        xy_mesh_batch = np.random.rand(batchsize_it, dim_size)

        u = sess.run(U_hat, feed_dict={XY_mesh: xy_mesh_batch})
        print(u)


def test_fun2():
    hidden_layer = (2, 3, 4, 5, 6)
    batchsize_it = 100
    dim_size = 2
    input_dim = 2
    out_dim = 1
    act_func = tf.nn.relu
    weights2kernel, bias2kernel = construct_weights2kernel(
        height2kernel=2, width2kernel=3, hidden_layers=hidden_layer, in_channel=input_dim, out_channel=out_dim)
    with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
        XY_it = tf.placeholder(tf.float32, name='X_it', shape=[batchsize_it, dim_size])
        adi_mat = pairwise_distance(XY_it)
        idx2nn = knn(adi_mat, k=10)
        U_hat = PDE_SingleGNN_myConv(XY_it, weight=weights2kernel[0], bias=bias2kernel[0], nn_idx=idx2nn, k_neighbors=10,
                                     act_name2stretch='relu', trans_dim=True, act_func2neigh_atten=act_func)
        U_hat = tf.reduce_mean(U_hat, axis=-1)
        U_hat = tf.squeeze(U_hat)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        xy_mesh_batch = np.random.rand(batchsize_it, dim_size)


if __name__ == "__main__":
    test_fun1()

