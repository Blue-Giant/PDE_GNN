# coding:utf-8
# author: xi'an Li
# date: 2020.12.12

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


def gauss(x):
    return tf.exp(-0.5*x*x)


def mexican(x):
    return (1-x*x)*tf.exp(-0.5*x*x)


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
def Generally_Init_Kernel(height2kernel=1, width2kernel=1, in_size=2, out_size=1, hidden_layers=None, Flag=None):
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


def Truncated_normal_init_Kernel(height2kernel=1, width2kernel=1, in_size=2, out_size=1, hidden_layers=None, Flag=None):
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


def Rand_Normal_init_Kernel(height2kernel=1, width2kernel=1, in_size=2, out_size=1, hidden_layers=None,
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


def Xavier_init_Kernel(height2kernel=1, width2kernel=1, in_size=2, out_size=1, hidden_layers=None, Flag=None, varcoe=0.5):
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


def Xavier_init_Fourier_Kernel(height2kernel=1, width2kernel=1, in_size=2, out_size=1, hidden_layers=None, Flag=None, varcoe=0.5):
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
            if i_layer == 0:
                stddev_KB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
                W = tf.get_variable(name='Kernel' + str(i_layer + 1) + str(Flag),
                                    shape=(height2kernel, width2kernel, 2*hidden_layers[i_layer], hidden_layers[i_layer + 1]),
                                    initializer=tf.random_normal_initializer(stddev=stddev_KB),
                                    dtype=tf.float32)
                B = tf.get_variable(name='B-hidden' + str(i_layer + 1) + str(Flag),
                                    shape=(1, 1, 1, hidden_layers[i_layer + 1]),
                                    initializer=tf.random_normal_initializer(stddev=stddev_KB),
                                    dtype=tf.float32)
            else:
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


def Xavier_init_attenKernel(height2kernel=1, width2kernel=1, in_size=2, out_size=1, kneigh=4, hidden_layers=None, Flag=None, varcoe=0.5):
    with tf.variable_scope('Kernel_scope', reuse=tf.AUTO_REUSE):
        n_hiddens = len(hidden_layers)
        kernels = []  # Kernel列表，用于存储隐藏层的权重
        Biases = []  # 偏置列表，用于存储隐藏层的偏置
        # 隐藏层：第一层的kernel，对输入数据做变换
        stddev_KB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.get_variable(name='Kernel-transInput' + str(Flag),
                            shape=(height2kernel, width2kernel, hidden_layers[0], 1),
                            initializer=tf.random_normal_initializer(stddev=stddev_KB),
                            dtype=tf.float32)
        B = tf.get_variable(name='B-transInput' + str(Flag),
                            shape=(1, 1, kneigh, 1),
                            initializer=tf.random_normal_initializer(stddev=stddev_KB),
                            dtype=tf.float32)
        kernels.append(W)
        Biases.append(B)
        for i_layer in range(0, n_hiddens - 1):
            stddev_KB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
            W = tf.get_variable(name='Kernel' + str(i_layer + 1) + str(Flag),
                                shape=(height2kernel, width2kernel, hidden_layers[i_layer + 1], 1),
                                initializer=tf.random_normal_initializer(stddev=stddev_KB),
                                dtype=tf.float32)
            B = tf.get_variable(name='B-hidden' + str(i_layer + 1) + str(Flag),
                                shape=(1, 1, kneigh, 1),
                                initializer=tf.random_normal_initializer(stddev=stddev_KB),
                                dtype=tf.float32)
            kernels.append(W)
            Biases.append(B)
        # 输出层：最后一层B的kernel。将最后的结果变换到输出维度
        stddev_KB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.get_variable(name='Kernel-outTrans' + str(Flag),
                            shape=(height2kernel, width2kernel, hidden_layers[-1], 1),
                            initializer=tf.random_normal_initializer(stddev=stddev_KB),
                            dtype=tf.float32)
        B = tf.get_variable(name='B-outTrans' + str(Flag),
                            shape=(1, 1, kneigh, 1),
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


# ----------------------------------- 正则化 -----------------------------------------------
def regular_weights_biases_L1(weights, biases):
    # L1正则化权重和偏置
    layers = len(weights)
    regular_w = 0
    regular_b = 0
    for i_layer1 in range(layers):
        regular_w = regular_w + tf.reduce_sum(tf.abs(weights[i_layer1]), keep_dims=False)
        regular_b = regular_b + tf.reduce_sum(tf.abs(biases[i_layer1]), keep_dims=False)
    return regular_w + regular_b


# L2正则化权重和偏置
def regular_weights_biases_L2(weights, biases):
    layers = len(weights)
    regular_w = 0
    regular_b = 0
    for i_layer1 in range(layers):
        regular_w = regular_w + tf.reduce_sum(tf.square(weights[i_layer1]), keep_dims=False)
        regular_b = regular_b + tf.reduce_sum(tf.square(biases[i_layer1]), keep_dims=False)
    return regular_w + regular_b


def my_conv2d(v_input, kernel=None, bias2conv=None, actName=None, if_add_bias=True):
    """Construct new-point feature for each point
        Args:
        v_input: (num_points, num_dims) or (1, 1, num_points, num_dims)
        kernel: (num_dims, new_num_dims) or (1, *, num_points, new_num_dims)
        bias2conv: (1, 1, 1, new_num_dims)
        actName: the name of activation function

        Returns:
        v_output: (1,1, num_points, new_num_dims)
    """
    if actName == 'relu':
        GNN_activation = tf.nn.relu
    elif actName == 'leaky_relu':
        GNN_activation = tf.nn.leaky_relu
    elif actName == 'srelu':
        GNN_activation = srelu
    elif actName == 's2relu':
        GNN_activation = s2relu
    elif actName == 'elu':
        GNN_activation = tf.nn.elu
    elif actName == 'sin':
        GNN_activation = mysin
    elif actName == 'tanh':
        GNN_activation = tf.nn.tanh
    elif actName == 'gauss':
        GNN_activation = gauss
    elif actName == 'mexican':
        GNN_activation = mexican
    elif actName == 'phi':
        GNN_activation = phi

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
    if if_add_bias:
        out_result = GNN_activation(tf.add(outputs, bias2conv))
    else:
        out_result = GNN_activation(outputs)

    return out_result


def myConv2d_no_activate(v_input, kernel=None, bias2conv=None, if_scale=False, scale_array=None, add_bias=True):
    """Construct new-point feature for each point
    How to use tf.nn.conv2D(): https://blog.csdn.net/zuolixiangfisher/article/details/80528989
                               https://blog.csdn.net/qq_35203425/article/details/81193293
        Args:
        v_input: (num_points, dim) or (1, 1, num_points, dim)
        kernel: (num_dims, new_dim) or (1, *, num_points, new_dim)
        bias2conv: (1, 1, 1, new_dim)
        if_scale: bool -- whether need scale transform for input
        scale_array: the array of scale factors

        Returns:
        v_output: (1,1, num_points, new_dim)
    """
    shape2v = v_input.get_shape()
    if len(shape2v) == 2:
        # expand dim in the first index for input(two times)
        v_input = tf.expand_dims(v_input, axis=0)
        v_input = tf.expand_dims(v_input, axis=0)
    elif len(shape2v) == 3:
        # expand dim in the first index for input(one time)
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
        Unit_num = int(shape2out[-1] // len(scale_array))

        # np.repeat(a, repeats, axis=None)
        # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
        # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
        mixcoe = np.repeat(scale_array, Unit_num)

        # 这个的作用是什么？
        mixcoe = np.concatenate((mixcoe, np.ones([shape2out[-1] - Unit_num * len(scale_array)]) * scale_array[-1]))

        mixcoe = mixcoe.astype(np.float32)
        mixcoe = np.expand_dims(mixcoe, axis=0)
        mixcoe = np.expand_dims(mixcoe, axis=0)
        mixcoe = np.expand_dims(mixcoe, axis=0)
        outputs = tf.multiply(outputs, mixcoe)
    if add_bias:
        out_result = tf.add(outputs, bias2conv)
    else:
        out_result = outputs

    return out_result


def pairwise_distance(point_set):
    """Compute pairwise distance of a point cloud.
        Args:
          (x-y)^2 = x^2 - 2xy + y^2
          point_set: tensor (num_points, dims2point)
        Returns:
          pairwise distance: (num_points, num_points)
    """
    point_set_shape = point_set.get_shape().as_list()
    assert(len(point_set_shape)) == 2

    point_set_transpose = tf.transpose(point_set, perm=[1, 0])
    point_set_inner = tf.matmul(point_set, point_set_transpose)
    point_set_inner = -2 * point_set_inner
    point_set_square = tf.reduce_sum(tf.square(point_set), axis=-1, keep_dims=True)
    point_set_square_transpose = tf.transpose(point_set_square, perm=[1, 0])
    return point_set_square + point_set_inner + point_set_square_transpose


def knn_includeself(dist_matrix, k=20):
    """Get KNN based on the pairwise distance.
        How to use tf.nn.top_k(): https://blog.csdn.net/wuguangbin1230/article/details/72820627
      Args:
        pairwise distance: (num_points, num_points)
        k: int

      Returns:
        nearest neighbors: (num_points, k)
      """
    neg_dist = -1.0*dist_matrix
    _, nn_idx = tf.nn.top_k(neg_dist, k=k)  # 这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
    return nn_idx


def knn_excludeself(dist_matrix, k=20):
    """Get KNN based on the pairwise distance.
      Args:
        pairwise distance: (num_points, num_points)
        k: int

      Returns:
        nearest neighbors: (num_points, k)
      """
    neg_dist = -1.0*dist_matrix
    k_neighbors = k+1
    _, knn_idx = tf.nn.top_k(neg_dist, k=k_neighbors)  # 这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
    nn_idx = knn_idx[:, 1: k_neighbors]
    return nn_idx


def get_kneighbors(point_set, nn_idx):
    """Construct neighbors feature for each point
        Args:
        point_set: (batch_size, num_points, 1, dim)
        nn_idx: (batch_size, num_points, k)
        k: int

        Returns:
        neighbors features: (batch_size, num_points, k, dim)
      """
    og_batch_size = point_set.get_shape().as_list()[0]
    og_num_dims = point_set.get_shape().as_list()[-1]
    point_set = tf.squeeze(point_set)
    if og_batch_size == 1:
        point_set = tf.expand_dims(point_set, 0)
    if og_num_dims == 1:
        point_set = tf.expand_dims(point_set, -1)

    point_set_shape = point_set.get_shape()
    batch_size = point_set_shape[0].value
    num_points = point_set_shape[1].value
    num_dims = point_set_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_set_flat = tf.reshape(point_set, [-1, num_dims])
    point_set_neighbors = tf.gather(point_set_flat, nn_idx + idx_)

    return point_set_neighbors


# 对于每个点，半径内的点数不一样，如何搞？后续再研究
def get_radius_neighbors_inclueself(point_set, radius=0.5):
    """Construct neighbors feature for each point
        Args:
        point_set: (batch_size, num_points, 1, num_dims)
        radius: float

        Returns:
        neighbors features: (batch_size, num_points, k, num_dims)
      """
    dis_matrix = pairwise_distance(point_set)
    condition = tf.less(dis_matrix, radius*radius)
    bool_idx = tf.equal(condition, True)
    nn_idx = tf.where(bool_idx)
    og_batch_size = point_set.get_shape().as_list()[0]
    og_num_dims = point_set.get_shape().as_list()[-1]
    point_set = tf.squeeze(point_set)
    if og_batch_size == 1:
        point_set = tf.expand_dims(point_set, 0)
    if og_num_dims == 1:
        point_set = tf.expand_dims(point_set, -1)

    point_set_shape = point_set.get_shape()
    batch_size = point_set_shape[0].value
    num_points = point_set_shape[1].value
    num_dims = point_set_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_set_flat = tf.reshape(point_set, [-1, num_dims])
    point_set_neighbors = tf.gather(point_set_flat, nn_idx + idx_)

    return point_set_neighbors


def cal_attens2neighbors(edge_point_set):
    square_edges = tf.square(edge_point_set)
    norm2edges = tf.reduce_sum(square_edges, axis=-1)
    exp_dis = tf.exp(-norm2edges)
    expand_exp_dis = tf.expand_dims(exp_dis, axis=-2)
    return expand_exp_dis


# 利用输入的卷积核和偏置进行卷积运算，卷积函数使用本人自定义的
def SingleGNN_myConv(point_set, weight=None, bias=None, nn_idx=None, k_neighbors=10, activate_name='tanh', freqs=None,
                     opt2cal_atten='dist_attention', actName2atten='relu', kernel2atten=None, bias2atten=None):
    """Construct edge feature for each point
        Args:
        point_set: float array -- (num_points, in_dim)
        weight: (1, 1, in_dim, out_dim)
        bias:  (1, 1, 1, out_dim)
        nn_idx: int array --(num_points, k)
        k_neighbors: int
        activate_name: string -- the name of activation function for changing the dimension of input-variable
        freqs: array -- the array of scale factors
        opt2cal_atten: the option for calculating the attention coefficient of neighbors
        actName2atten: the activation function of obtaining the coefficient for different neighbor point
        kernel2atten: (1, 1, out_dim, 1)
        bias2atten: (1, 1, k_neighbors, 1)

        Returns:
        new point_set: (num_points, out_dim)
    """
    if activate_name == 'relu':
        GNN_activation = tf.nn.relu
    elif activate_name == 'leaky_relu':
        GNN_activation = tf.nn.leaky_relu
    elif activate_name == 'srelu':
        GNN_activation = srelu
    elif activate_name == 's2relu':
        GNN_activation = s2relu
    elif activate_name == 'elu':
        GNN_activation = tf.nn.elu
    elif activate_name == 'sin':
        GNN_activation = mysin
    elif activate_name == 'tanh':
        GNN_activation = tf.nn.tanh
    elif activate_name == 'gauss':
        GNN_activation = gauss
    elif activate_name == 'mexican':
        GNN_activation = mexican
    elif activate_name == 'phi':
        GNN_activation = phi
    point_set_shape = point_set.get_shape()
    assert (len(point_set_shape)) == 2
    num_points = point_set_shape[0].value
    num_dims = point_set_shape[1].value

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

    expand_point_set = tf.expand_dims(tf.expand_dims(point_set, axis=0), axis=-2)
    new_points_cloud = myConv2d_no_activate(expand_point_set, kernel=weight, bias2conv=bias, if_scale=False,
                                            scale_array=freqs, add_bias=True)
    squeeze_new_points = tf.squeeze(new_points_cloud)

    # 选出每个点的邻居，然后的到每个点和各自邻居点的边
    select_idx = nn_idx                                            # 邻居点的indexes
    point_neighbors = tf.gather(squeeze_new_points, select_idx)    # 邻居点的'坐标' (num_points, k_neighbors, dim2point)

    point_central = tf.expand_dims(squeeze_new_points, axis=-2)    # 每个点作为中心点，在倒数第二个位置扩维
    centroids_tilde = tf.tile(point_central, [1, k_neighbors, 1])  # 中心点复制k份
    edges_feature = centroids_tilde - point_neighbors              # 中心点和各自的邻居点形成的边

    # 根据边的长度计算每个邻居点对应的权重系数，然后根据系数计算邻居点的边对中心点的贡献
    if opt2cal_atten == 'dist_attention':
        atten_ceof2neighbors = cal_attens2neighbors(edges_feature)
        atten_ceof2neighbors = tf.nn.softmax(atten_ceof2neighbors)
        atten_neighbors = tf.matmul(atten_ceof2neighbors, edges_feature)  # 利用注意力系数将各个邻居聚合起来
    elif opt2cal_atten == 'conv_attention':
        expand_edges = tf.expand_dims(edges_feature, axis=0)
        atten_ceof2neighbors = my_conv2d(expand_edges, kernel=kernel2atten, bias2conv=bias2atten, actName=actName2atten)

        atten_ceof2neighbors = tf.nn.softmax(atten_ceof2neighbors)
        neighbors_coef = tf.transpose(atten_ceof2neighbors, perm=[0, 1, 3, 2])

        expand_neighbors = tf.expand_dims(point_neighbors, axis=0)
        atten_neighbors = tf.matmul(neighbors_coef, expand_neighbors)   # 利用注意力系数将各个邻居聚合起来

    squeeze2atten_neighbors = tf.squeeze(atten_neighbors)                   # 去除数值为1的维度

    # 中心点的特征和邻居点的特征相加，得到新的特征
    new_point_set = GNN_activation(tf.add(squeeze_new_points, squeeze2atten_neighbors))

    return new_point_set


# 利用输入的卷积核和偏置进行卷积运算，卷积函数使用本人自定义的
def HierarchicGNN_myConv(point_set, Weight_list=None, Bias_list=None, nn_idx=None, k_neighbors=10, activate_name=None,
                         scale_trans=False, freqs=None, opt2cal_atten='dist_attention', actName2atten='relu',
                         kernels2atten=None, biases2atten=None, hiddens=None):
    """Construct edge feature for each point
        Args:
        point_set: float array -- (num_points, num_dims)
        Weight_list: (1, 1, num_dim, out_dim1), (1, 1, out_dim1, out_dim2),.....
        Bias_list: (1, 1, 1, out_dim1), (1, 1, 1, out_dim2),.....
        nn_idx: int array --(num_points, k)
        k_neighbors: int
        activate_name: string -- the name of activation function for  changing the dimension of input-variable
        scale_trans: bool
        freqs: the array of scale factors
        opt2cal_atten：string, the option for calculating the attention coefficient of neighbors
        actName2atten: the activation function of obtaining the coefficient for different neighbor point
        kernels2atten:(1, 1, hiddens[0], 1), (1, 1, hiddens[1], 1),(1, 1, hiddens[2], 1),...
        biases2atten: (1, 1, k_neighbors, 1),(1, 1, k_neighbors, 1), (1, 1, k_neighbors, 1),...
        hiddens: a list --- the number of unit for various hidden layers

        Returns:
        new point_set: (num_points, out_dim)
    """
    if activate_name == 'relu':
        GNN_activation = tf.nn.relu
    elif activate_name == 'leaky_relu':
        GNN_activation = tf.nn.leaky_relu
    elif activate_name == 'srelu':
        GNN_activation = srelu
    elif activate_name == 's2relu':
        GNN_activation = s2relu
    elif activate_name == 'elu':
        GNN_activation = tf.nn.elu
    elif activate_name == 'sin':
        GNN_activation = mysin
    elif activate_name == 'tanh':
        GNN_activation = tf.nn.tanh
    elif activate_name == 'gauss':
        GNN_activation = gauss
    elif activate_name == 'mexican':
        GNN_activation = mexican
    elif activate_name == 'phi':
        GNN_activation = phi

    point_set_shape = point_set.get_shape()
    assert (len(point_set_shape)) == 2
    num_points = point_set_shape[0].value
    num_dims = point_set_shape[1].value

    # select_idx = nn_idx + num_points  # 邻居点的indexes
    select_idx = nn_idx                 # 邻居点的indexes

    out_set = point_set
    layers = len(hiddens)
    hidden_record = 0
    for i_layer in range(layers):
        out_pre = out_set
        weight = Weight_list[i_layer]
        bias = Bias_list[i_layer]
        kernel2atten = kernels2atten[i_layer]
        bias2atten = biases2atten[i_layer]
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
        new_points_cloud = myConv2d_no_activate(out_set, kernel=weight, bias2conv=bias, if_scale=scale_trans,
                                                scale_array=freqs, add_bias=True)
        squeeze_new_points = tf.squeeze(tf.squeeze(new_points_cloud, axis=0), axis=0)
        assert len(squeeze_new_points.get_shape()) == 2

        # 选出每个点的邻居，然后的到每个点和各自邻居点的边
        point_neighbors = tf.gather(squeeze_new_points, select_idx)    # 邻居点的'坐标'

        point_central = tf.expand_dims(squeeze_new_points, axis=-2)    # 每个点作为中心点，在倒数第二个位置扩维
        centroids_tilde = tf.tile(point_central, [1, k_neighbors, 1])  # 中心点复制k份
        edges_feature = centroids_tilde - point_neighbors              # 中心点和各自的邻居点形成的边(num_points,k,dim)

        # 根据边的长度计算每个邻居点对应的权重系数，然后根据系数计算邻居点对中心点的贡献
        if opt2cal_atten == 'dist_attention':
            atten_ceof2neighbors = cal_attens2neighbors(edges_feature)
            atten_ceof2neighbors = tf.nn.softmax(atten_ceof2neighbors)
            atten_neighbors = tf.matmul(atten_ceof2neighbors, edges_feature)  # 利用注意力系数将各个邻居聚合起来
            squeeze2atten_neighbors = tf.squeeze(atten_neighbors, axis=1)     # 去除数值为1的维度
        else:
            # conv2d 函数接收的输入是一个 4d tensor，先将edges_feature扩维
            expand_edges = tf.expand_dims(edges_feature, axis=0)    # (1, num_points, k, dim)
            atten_ceof2neighbors = my_conv2d(expand_edges, kernel=kernel2atten, bias2conv=bias2atten, actName=actName2atten)
            # 归一化得到注意力系数
            atten_ceof2neighbors = tf.nn.softmax(atten_ceof2neighbors)
            neighbors_coef = tf.transpose(atten_ceof2neighbors, perm=[0, 1, 3, 2])

            expand_neighbors = tf.expand_dims(point_neighbors, axis=0)
            atten_neighbors = tf.matmul(neighbors_coef, expand_neighbors)  # 利用注意力系数将各个邻居聚合起来

            squeeze2atten_neighbors = tf.squeeze(tf.squeeze(atten_neighbors, axis=0), axis=1)     # 去除数值为1的维度
        # 中心点的特征和邻居点的特征相加，得到新的特征
        out_set = GNN_activation(tf.add(squeeze_new_points, squeeze2atten_neighbors))
        if hiddens[i_layer] == hidden_record:
            out_set = tf.add(out_set, out_pre)
        hidden_record = hiddens[i_layer]

        scale_trans = False

    return out_set


# 利用输入的卷积核和偏置进行卷积运算，卷积函数使用本人自定义的
def FourierHierarchicGNN_myConv(point_set, Weight_list=None, Bias_list=None, nn_idx=None, k_neighbors=5,
                                activate_name=None, scale_trans=False, freqs=None,  opt2cal_atten='dist_attention',
                                actName2atten='relu', kernels2atten=None, biases2atten=None, hiddens=None, sFourier=1.0):
    """Construct edge feature for each point
        Args:
        point_set: float array -- (num_points, num_dims)
        Weight_list: (1, 1, num_dim, out_dim1), (1, 1, out_dim1, out_dim2),.....
        Bias_list: (1, 1, 1, out_dim1), (1, 1, 1, out_dim2),.....
        nn_idx: int array --(num_points, k)
        k_neighbors: int
        activate_name: string -- the name of activation function for  changing the dimension of input-variable
        scale_trans: bool
        freqs: the array of scale factors
        opt2cal_atten：string, the option for calculating the attention coefficient of neighbors
        actName2atten: the activation function of obtaining the coefficient for different neighbor point
        kernels2atten:(1, 1, hiddens[0], 1), (1, 1, hiddens[1], 1),(1, 1, hiddens[2], 1),...
        biases2atten: (1, 1, k_neighbors, 1),(1, 1, k_neighbors, 1), (1, 1, k_neighbors, 1),...
        hiddens: a list --- the number of unit for various hidden layers

        Returns:
        new point_set: (num_points, out_dim)
    """
    if activate_name == 'relu':
        GNN_activation = tf.nn.relu
    elif activate_name == 'leaky_relu':
        GNN_activation = tf.nn.leaky_relu
    elif activate_name == 'srelu':
        GNN_activation = srelu
    elif activate_name == 's2relu':
        GNN_activation = s2relu
    elif activate_name == 'elu':
        GNN_activation = tf.nn.elu
    elif activate_name == 'sin':
        GNN_activation = mysin
    elif activate_name == 'tanh':
        GNN_activation = tf.nn.tanh
    elif activate_name == 'gauss':
        GNN_activation = gauss
    elif activate_name == 'mexican':
        GNN_activation = mexican
    elif activate_name == 'phi':
        GNN_activation = phi

    point_set_shape = point_set.get_shape()
    assert (len(point_set_shape)) == 2
    num_points = point_set_shape[0].value
    num_dims = point_set_shape[1].value

    # select_idx = nn_idx + num_points  # 邻居点的indexes
    select_idx = nn_idx                 # 邻居点的indexes

    out_set = point_set

    layers = len(hiddens)
    hidden_record = 0
    for i_layer in range(layers):
        out_pre = out_set
        weight = Weight_list[i_layer]
        bias = Bias_list[i_layer]
        kernel2atten = kernels2atten[i_layer]
        bias2atten = biases2atten[i_layer]
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
        new_points_cloud = myConv2d_no_activate(out_set, kernel=weight, bias2conv=bias, if_scale=scale_trans,
                                                scale_array=freqs)
        squeeze_new_points = tf.squeeze(tf.squeeze(new_points_cloud, axis=0), axis=0)
        assert len(squeeze_new_points.get_shape()) == 2

        # 选出每个点的邻居，然后的到每个点和各自邻居点的边
        point_neighbors = tf.gather(squeeze_new_points, select_idx)    # 邻居点的'坐标'

        point_central = tf.expand_dims(squeeze_new_points, axis=-2)    # 每个点作为中心点，在倒数第二个位置扩维
        centroids_tilde = tf.tile(point_central, [1, k_neighbors, 1])  # 中心点复制k份
        edges_feature = centroids_tilde - point_neighbors              # 中心点和各自的邻居点形成的边(num_points,k,dim)

        # 根据边的长度计算每个邻居点对应的权重系数，然后根据系数计算邻居点对中心点的贡献
        if opt2cal_atten == 'dist_attention':
            atten_ceof2neighbors = cal_attens2neighbors(edges_feature)
            atten_ceof2neighbors = tf.nn.softmax(atten_ceof2neighbors)
            atten_neighbors = tf.matmul(atten_ceof2neighbors, edges_feature)  # 利用注意力系数将各个邻居聚合起来
            squeeze2atten_neighbors = tf.squeeze(atten_neighbors, axis=1)     # 去除数值为1的维度
        else:
            # conv2d 函数接收的输入是一个 4d tensor，先将edges_feature扩维
            expand_edges = tf.expand_dims(edges_feature, axis=0)    # (1, num_points, k, dim)
            atten_ceof2neighbors = my_conv2d(expand_edges, kernel=kernel2atten, bias2conv=bias2atten, actName=actName2atten)
            # 归一化得到注意力系数
            atten_ceof2neighbors = tf.nn.softmax(atten_ceof2neighbors)
            neighbors_coef = tf.transpose(atten_ceof2neighbors, perm=[0, 1, 3, 2])

            expand_neighbors = tf.expand_dims(point_neighbors, axis=0)
            atten_neighbors = tf.matmul(neighbors_coef, expand_neighbors)  # 利用注意力系数将各个邻居聚合起来

            squeeze2atten_neighbors = tf.squeeze(tf.squeeze(atten_neighbors, axis=0), axis=1)     # 去除数值为1的维度
        # 中心点的特征和邻居点的特征相加，得到新的特征
        out_set = GNN_activation(tf.add(squeeze_new_points, squeeze2atten_neighbors))
        if i_layer == 0:
            out_set = sFourier*tf.concat([tf.cos(out_set), tf.cos(out_set)], axis=-1)
        else:
            if hiddens[i_layer] == hidden_record:
                out_set = tf.add(out_set, out_pre)
        hidden_record = hiddens[i_layer]
        scale_trans = False

    return out_set


def CNN_model(v_input, kernels=None, act_function=tf.nn.relu):
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


def test_CNN_model():
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
        U_hat = CNN_model(XY_mesh, kernels=weights2kernel, act_function=activate_func)
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


def test_SingleGNN_myConv():
    height2kernel = 1
    width2kernel = 1
    batchsize_it = 10
    input_dim = 2
    out_dim = 8
    knn2xy = 4
    freq = np.array([1, 2, 3])
    kernel = weight_variable([height2kernel, width2kernel, int(input_dim), out_dim], name2weight='W_in')
    bias = weight_variable([height2kernel, width2kernel, 1, out_dim], name2weight='B_in')
    kernel_atten = weight_variable([height2kernel, width2kernel, out_dim, 1], name2weight='Watten')
    bias_atten = weight_variable([height2kernel, width2kernel, knn2xy, 1], name2weight='Batten')
    with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
        XY_mesh = tf.placeholder(tf.float32, name='X_it', shape=[batchsize_it, input_dim])
        # 在变量的内部区域训练
        adj_matrix = pairwise_distance(XY_mesh)
        idx2knn = knn_excludeself(adj_matrix, k=knn2xy)
        U_hat = SingleGNN_myConv(XY_mesh, weight=kernel, bias=bias, nn_idx=idx2knn, k_neighbors=knn2xy,
                                 activate_name='relu', freqs=freq, opt2cal_atten='conv_attention',
                                 kernel2atten=kernel_atten, bias2atten=bias_atten)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        xy_mesh_batch = np.random.rand(batchsize_it, input_dim)

        u = sess.run(U_hat, feed_dict={XY_mesh: xy_mesh_batch})
        print(u)


def test_HierarchicGNN_myConv():
    h2kernel = 1
    w2kernel = 1
    batchsize_it = 10
    input_dim = 2
    out_dim = 5
    knn2xy = 4
    activate_func = tf.nn.relu
    hidden_list = (2, 3, 4, 5, 6)
    freq = np.array([1, 2, 3])
    flag2kernel = 'WB'
    W2NN, B2NN = Xavier_init_Kernel(height2kernel=h2kernel, width2kernel=w2kernel, in_size=input_dim,
                                    out_size=out_dim, hidden_layers=hidden_list, Flag=flag2kernel)

    flag2attenKernel = 'Kernel2atten'
    W2atten, B2atten = Xavier_init_attenKernel(height2kernel=h2kernel, width2kernel=w2kernel,
                                               in_size=input_dim, out_size=out_dim, kneigh=knn2xy,
                                               hidden_layers=hidden_list, Flag=flag2attenKernel)
    with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
        XY_mesh = tf.placeholder(tf.float32, name='X_it', shape=[batchsize_it, input_dim])
        # 在变量的内部区域训练
        adj_matrix = pairwise_distance(XY_mesh)
        idx2knn = knn_excludeself(adj_matrix, k=knn2xy)
        U_hat = HierarchicGNN_myConv(XY_mesh, Weight_list=W2NN, Bias_list=B2NN, nn_idx=idx2knn, k_neighbors=knn2xy,
                                     activate_name='relu', scale_trans=True, freqs=freq, opt2cal_atten='conv_attention',
                                     actName2atten='relu', kernels2atten=W2atten, biases2atten=B2atten,
                                     hiddens=hidden_list)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        xy_mesh_batch = np.random.rand(batchsize_it, input_dim)

        u = sess.run(U_hat, feed_dict={XY_mesh: xy_mesh_batch})
        print(u)


def test_FourierHierarchicGNN_myConv():
    h2kernel = 1
    w2kernel = 1
    batchsize_it = 10
    input_dim = 2
    out_dim = 5
    knn2xy = 4
    activate_func = tf.nn.relu
    hidden_list = (2, 3, 4, 5, 6)
    freq = np.array([1, 2, 3])
    flag2kernel = 'WB'
    W2NN, B2NN = Xavier_init_Fourier_Kernel(height2kernel=h2kernel, width2kernel=w2kernel, in_size=input_dim,
                                            out_size=out_dim, hidden_layers=hidden_list, Flag=flag2kernel)
    with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
        XY_mesh = tf.placeholder(tf.float32, name='X_it', shape=[batchsize_it, input_dim])
        # 在变量的内部区域训练
        adj_matrix = pairwise_distance(XY_mesh)
        idx2knn = knn_excludeself(adj_matrix, k=knn2xy)
        U_hat = FourierHierarchicGNN_myConv(XY_mesh, Weight_list=W2NN, Bias_list=B2NN, nn_idx=idx2knn,
                                            k_neighbors=knn2xy, activate_name='relu', scale_trans=True, freqs=freq,
                                            opt2cal_atten='my_cal_attention', hiddens=hidden_list, sFourier=1.0)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        xy_mesh_batch = np.random.rand(batchsize_it, input_dim)

        u = sess.run(U_hat, feed_dict={XY_mesh: xy_mesh_batch})
        print(u)


if __name__ == "__main__":
    # test_CNN_model()
    # test_SingleGNN_myConv()
    test_HierarchicGNN_myConv()
    # test_FourierHierarchicGNN_myConv()

