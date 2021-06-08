# coding:utf-8
# author: xi'an Li
# date: 2020.12.12

import tensorflow as tf
import numpy as np


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


# ------------------------------------------------  初始化权重和偏置 --------------------------------------------
# 生成DNN的权重和偏置
# tf.random_normal(): 用于从服从指定正太分布的数值中取出随机数
# tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
# hape: 输出张量的形状，必选.--- mean: 正态分布的均值，默认为0.----stddev: 正态分布的标准差，默认为1.0
# dtype: 输出的类型，默认为tf.float32 ----seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样---name: 操作的名称
def Generally_Init_NN(in_size, out_size, hidden_layers, Flag='flag'):
    n_hiddens = len(hidden_layers)
    Weights = []  # 权重列表，用于存储隐藏层的权重
    Biases = []  # 偏置列表，用于存储隐藏层的偏置
    # 隐藏层：第一层的权重和偏置，对输入数据做变换
    W = tf.Variable(0.1 * tf.random.normal([in_size, hidden_layers[0]]), dtype='float32',
                    name='W_transInput' + str(Flag))
    B = tf.Variable(0.1 * tf.random.uniform([1, hidden_layers[0]]), dtype='float32',
                    name='B_transInput' + str(Flag))
    Weights.append(W)
    Biases.append(B)
    # 隐藏层：第二至倒数第二层的权重和偏置
    for i_layer in range(n_hiddens - 1):
        W = tf.Variable(0.1 * tf.random.normal([hidden_layers[i_layer], hidden_layers[i_layer+1]]), dtype='float32',
                        name='W_hidden' + str(i_layer + 1) + str(Flag))
        B = tf.Variable(0.1 * tf.random.uniform([1, hidden_layers[i_layer+1]]), dtype='float32',
                        name='B_hidden' + str(i_layer + 1) + str(Flag))
        Weights.append(W)
        Biases.append(B)

    # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
    W = tf.Variable(0.1 * tf.random.normal([hidden_layers[-1], out_size]), dtype='float32',
                    name='W_outTrans' + str(Flag))
    B = tf.Variable(0.1 * tf.random.uniform([1, out_size]), dtype='float32',
                    name='B_outTrans' + str(Flag))
    Weights.append(W)
    Biases.append(B)

    return Weights, Biases


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
def truncated_normal_init(in_dim, out_dim, scale_coef=1.0, weight_name='weight'):
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    # 尺度因子防止初始化的数值太小或者太大
    V = tf.Variable(scale_coef*tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32, name=weight_name)
    return V


# tf.random_uniform()
# 默认是在 0 到 1 之间产生随机数，也可以通过 minval 和 maxval 指定上下界
def uniform_init(in_dim, out_dim, weight_name='weight'):
    V = tf.Variable(tf.random_uniform([in_dim, out_dim], dtype=tf.float32), dtype=tf.float32, name=weight_name)
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
def normal_init(in_dim, out_dim, scale_coef=1.0, weight_name='weight'):
    stddev2normal = np.sqrt(2.0/(in_dim + out_dim))
    # 尺度因子防止初始化的数值太小或者太大
    V = tf.Variable(scale_coef*tf.random_normal([in_dim, out_dim], mean=0, stddev=stddev2normal, dtype=tf.float32),
                    dtype=tf.float32, name=weight_name)
    return V


def Truncated_normal_init_NN(in_size, out_size, hidden_layers, Flag='flag'):
    with tf.variable_scope('WB_scope', reuse=tf.AUTO_REUSE):
        scale = 5.0
        n_hiddens = len(hidden_layers)
        Weights = []                  # 权重列表，用于存储隐藏层的权重
        Biases = []                   # 偏置列表，用于存储隐藏层的偏置

        # 隐藏层：第一层的权重和偏置，对输入数据做变换
        W = truncated_normal_init(in_size, hidden_layers[0], scale_coef=scale, weight_name='W-transInput' + str(Flag))
        B = uniform_init(1, hidden_layers[0], weight_name='B-transInput' + str(Flag))
        Weights.append(W)
        Biases.append(B)
        for i_layer in range(0, n_hiddens - 1):
            W = truncated_normal_init(hidden_layers[i_layer], hidden_layers[i_layer + 1], scale_coef=scale,
                                      weight_name='W-hidden' + str(i_layer + 1) + str(Flag))
            B = uniform_init(1, hidden_layers[i_layer + 1], weight_name='B-hidden' + str(i_layer + 1) + str(Flag))
            Weights.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        W = truncated_normal_init(hidden_layers[-1], out_size, scale_coef=scale, weight_name='W-outTrans' + str(Flag))
        B = uniform_init(1, out_size, weight_name='B-outTrans' + str(Flag))
        Weights.append(W)
        Biases.append(B)
        return Weights, Biases


def Xavier_init_NN(in_size, out_size, hidden_layers, Flag='flag', varcoe=0.5):
    with tf.variable_scope('WB_scope', reuse=tf.AUTO_REUSE):
        n_hiddens = len(hidden_layers)
        Weights = []  # 权重列表，用于存储隐藏层的权重
        Biases = []  # 偏置列表，用于存储隐藏层的偏置
        # 隐藏层：第一层的权重和偏置，对输入数据做变换
        stddev_WB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.get_variable(name='W-transInput' + str(Flag), shape=(in_size, hidden_layers[0]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        B = tf.get_variable(name='B-transInput' + str(Flag), shape=(hidden_layers[0],),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        Weights.append(W)
        Biases.append(B)
        for i_layer in range(0, n_hiddens - 1):
            stddev_WB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
            W = tf.get_variable(
                name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer], hidden_layers[i_layer + 1]),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            B = tf.get_variable(name='B' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer + 1],),
                                initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            Weights.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        stddev_WB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.get_variable(name='W-outTrans' + str(Flag), shape=(hidden_layers[-1], out_size),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
        B = tf.get_variable(name='B-outTrans' + str(Flag), shape=(out_size,),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)

        Weights.append(W)
        Biases.append(B)
        return Weights, Biases


def Xavier_init_NN_Fourier(in_size, out_size, hidden_layers, Flag='flag', varcoe=0.5):
    with tf.variable_scope('WB_scope', reuse=tf.AUTO_REUSE):
        n_hiddens = len(hidden_layers)
        Weights = []  # 权重列表，用于存储隐藏层的权重
        Biases = []  # 偏置列表，用于存储隐藏层的偏置
        # 隐藏层：第一层的权重和偏置，对输入数据做变换
        stddev_WB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.get_variable(name='W-transInput' + str(Flag), shape=(in_size, hidden_layers[0]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        B = tf.get_variable(name='B-transInput' + str(Flag), shape=(hidden_layers[0],),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        Weights.append(W)
        Biases.append(B)

        for i_layer in range(0, n_hiddens - 1):
            stddev_WB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
            if 0 == i_layer:
                W = tf.get_variable(
                    name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer]*2, hidden_layers[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
                B = tf.get_variable(name='B' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer + 1],),
                                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            else:
                W = tf.get_variable(
                    name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer], hidden_layers[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
                B = tf.get_variable(name='B' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer + 1],),
                                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            Weights.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        stddev_WB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.get_variable(name='W-outTrans' + str(Flag), shape=(hidden_layers[-1], out_size),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
        B = tf.get_variable(name='B-outTrans' + str(Flag), shape=(out_size,),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)

        Weights.append(W)
        Biases.append(B)
        return Weights, Biases


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


#  --------------------------------------------  网络模型 ------------------------------------------------------
def DNN(variable_input, Weights, Biases, hiddens, activate_name=None):
    if str.lower(activate_name) == 'relu':
        DNN_activation = tf.nn.relu
    elif str.lower(activate_name) == 'leaky_relu':
        DNN_activation = tf.nn.leaky_relu
    elif str.lower(activate_name) == 'srelu':
        DNN_activation = srelu
    elif str.lower(activate_name) == 's2relu':
        DNN_activation = s2relu
    elif str.lower(activate_name) == 'elu':
        DNN_activation = tf.nn.elu
    elif str.lower(activate_name) == 'sin':
        DNN_activation = mysin
    elif str.lower(activate_name) == 'tanh':
        DNN_activation = tf.nn.tanh
    elif str.lower(activate_name) == 'gauss':
        DNN_activation = gauss
    elif str.lower(activate_name) == 'mexican':
        DNN_activation = mexican
    elif str.lower(activate_name) == 'phi':
        DNN_activation = phi

    layers = len(hiddens) + 1               # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    hidden_record = 0
    for k in range(layers-1):
        H_pre = H
        W = Weights[k]
        B = Biases[k]
        H = DNN_activation(tf.add(tf.matmul(H, W), B))
        if hiddens[k] == hidden_record:
            H = H+H_pre
        hidden_record = hiddens[k]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # 下面这个是输出层
    # output = tf.nn.tanh(output)
    return output


def Kernel_DNN(x_input, Weights, Biases, hiddens, activate_name=None):
    if str.lower(activate_name) == 'relu':
        DNN_activation = tf.nn.relu
    elif str.lower(activate_name) == 'leaky_relu':
        DNN_activation = tf.nn.leaky_relu
    elif str.lower(activate_name) == 'srelu':
        DNN_activation = srelu
    elif str.lower(activate_name) == 's2relu':
        DNN_activation = s2relu
    elif str.lower(activate_name) == 'elu':
        DNN_activation = tf.nn.elu
    elif str.lower(activate_name) == 'sin':
        DNN_activation = mysin
    elif str.lower(activate_name) == 'tanh':
        DNN_activation = tf.nn.tanh
    elif str.lower(activate_name) == 'gauss':
        DNN_activation = gauss
    elif str.lower(activate_name) == 'mexican':
        DNN_activation = mexican
    elif str.lower(activate_name) == 'phi':
        DNN_activation = phi

    x_input_shape = x_input.get_shape()
    assert (len(x_input_shape)) == 3
    num_points = x_input_shape[0].value
    num_dims = x_input_shape[1].value

    layers = len(hiddens) + 1        # 得到输入到输出的层数，即隐藏层层数
    H = x_input                      # 代表输入数据，即输入层
    hidden_record = 0
    for k in range(layers-1):
        H_pre = H
        W = Weights[k]
        B = Biases[k]
        H = DNN_activation(tf.add(tf.matmul(H, W), B))
        if hiddens[k] == hidden_record:
            H = H+H_pre
        hidden_record = hiddens[k]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # 下面这个是输出层
    # output = tf.nn.tanh(output)
    return output


def DNN_scale(variable_input, Weights, Biases, hiddens, freq_frag, activate_name=None, repeat_Highfreq=True):
    if str.lower(activate_name) == 'relu':
        DNN_activation = tf.nn.relu
    elif str.lower(activate_name) == 'leaky_relu':
        DNN_activation = tf.nn.leaky_relu
    elif str.lower(activate_name) == 'srelu':
        DNN_activation = srelu
    elif str.lower(activate_name) == 's2relu':
        DNN_activation = s2relu
    elif str.lower(activate_name) == 'elu':
        DNN_activation = tf.nn.elu
    elif str.lower(activate_name) == 'sin':
        DNN_activation = mysin
    elif str.lower(activate_name) == 'tanh':
        DNN_activation = tf.nn.tanh
    elif str.lower(activate_name) == 'gauss':
        DNN_activation = gauss
    elif str.lower(activate_name) == 'mexican':
        DNN_activation = mexican
    elif str.lower(activate_name) == 'phi':
        DNN_activation = phi

    Unit_num = int(hiddens[0] / len(freq_frag))

    # np.repeat(a, repeats, axis=None)
    # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
    # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
    mixcoe = np.repeat(freq_frag, Unit_num)

    # 这个的作用是什么？
    if repeat_Highfreq==True:
        mixcoe = np.concatenate((mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[-1]))
    else:
        mixcoe = np.concatenate((np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[0], mixcoe))

    mixcoe = mixcoe.astype(np.float32)

    layers = len(hiddens) + 1  # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    W_in = Weights[0]
    B_in = Biases[0]
    if len(freq_frag) == 1:
        H = tf.add(tf.matmul(H, W_in), B_in)
    else:
        H = tf.add(tf.matmul(H, W_in)*mixcoe, B_in)

    H = DNN_activation(H)

    hidden_record = hiddens[0]
    for k in range(layers-2):
        H_pre = H
        W = Weights[k+1]
        B = Biases[k+1]
        H = DNN_activation(tf.add(tf.matmul(H, W), B))
        if hiddens[k+1] == hidden_record:
            H = H + H_pre
        hidden_record = hiddens[k+1]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # 下面这个是输出层
    # output = tf.nn.tanh(output)
    return output


# FourierBase 代表 cos concatenate sin according to row（i.e. the number of sampling points）
def DNN_FourierBase(variable_input, Weights, Biases, hiddens, freq_frag, activate_name=None, repeat_Highfreq=True,
                    sFourier=1.0):
    if str.lower(activate_name) == 'relu':
        DNN_activation = tf.nn.relu
    elif str.lower(activate_name) == 'leaky_relu':
        DNN_activation = tf.nn.leaky_relu
    elif str.lower(activate_name) == 'srelu':
        DNN_activation = srelu
    elif str.lower(activate_name) == 's2relu':
        DNN_activation = s2relu
    elif str.lower(activate_name) == 'elu':
        DNN_activation = tf.nn.elu
    elif str.lower(activate_name) == 'sin':
        DNN_activation = mysin
    elif str.lower(activate_name) == 'tanh':
        DNN_activation = tf.nn.tanh
    elif str.lower(activate_name) == 'gauss':
        DNN_activation = gauss
    elif str.lower(activate_name) == 'mexican':
        DNN_activation = mexican
    elif str.lower(activate_name) == 'phi':
        DNN_activation = phi

    layers = len(hiddens) + 1                   # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                          # 代表输入数据，即输入层

    # 计算第一个隐藏单元和尺度标记的比例
    Unit_num = int(hiddens[0] / len(freq_frag))

    # 然后，频率标记按按照比例复制
    # np.repeat(a, repeats, axis=None)
    # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
    # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
    mixcoe = np.repeat(freq_frag, Unit_num)

    if repeat_Highfreq == True:
        # 如果第一个隐藏单元的长度大于复制后的频率标记，那就按照最大的频率在最后补齐
        mixcoe = np.concatenate((mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[-1]))
    else:
        mixcoe = np.concatenate((mixcoe, np.ones([hiddens[0] - Unit_num * len(freq_frag)]) * freq_frag[0]))

    mixcoe = mixcoe.astype(np.float32)

    W_in = Weights[0]
    B_in = Biases[0]
    if len(freq_frag) == 1:
        H = tf.add(tf.matmul(H, W_in), B_in)
    else:
        # H = tf.add(tf.matmul(H, W_in)*mixcoe, B_in)
        H = tf.matmul(H, W_in) * mixcoe

    if str.lower(activate_name) == 'tanh':
        sfactor = sFourier
    elif str.lower(activate_name) == 's2relu':
        sfactor = 0.5
    else:
        sfactor = sFourier

    H = sfactor * (tf.concat([tf.cos(H), tf.sin(H)], axis=1))                        # sfactor=0.5 效果好
    # H = sfactor * (tf.concat([tf.cos(np.pi * H), tf.sin(np.pi * H)], axis=1))
    # H = sfactor * tf.concat([tf.cos(2 * np.pi * H), tf.sin(2 * np.pi * H)], axis=1)

    hiddens_record = hiddens[0]
    for k in range(layers-2):
        H_pre = H
        W = Weights[k+1]
        B = Biases[k+1]
        W_shape = W.get_shape().as_list()
        H = DNN_activation(tf.add(tf.matmul(H, W), B))
        if (hiddens[k+1] == hiddens_record) and (W_shape[0] == hiddens_record):
            H = H + H_pre
        hiddens_record = hiddens[k+1]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # 下面这个是输出层
    # output = tf.nn.tanh(output)
    return output


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


def cal_attens2neighbors(edge_point_set):
    """
        Args:
        edge_point_set:(num_points, k_neighbors, dim2point)
        return:
        expand_exp_dis:(num_points, 1, k_neighbors)
    """
    square_edges = tf.square(edge_point_set)            # (num_points, k_neighbors, dim2point)
    norm2edges = tf.reduce_sum(square_edges, axis=-1)   # (num_points, k_neighbors)
    exp_dis = tf.exp(-norm2edges)                       # (num_points, k_neighbors)
    expand_exp_dis = tf.expand_dims(exp_dis, axis=-2)   # (num_points, 1, k_neighbors)
    return expand_exp_dis


# single layer GKN module
def SingleGKN(point_set, coef_set=None, Ws2trans_input=None, Bs2trans_input=None, actName2trans_input='tanh',
              hiddens2trans_input=None, scale_factor=None, nn_idx=None, k_neighbors=10, Ws2kernel=None, Bs2kernel=None,
              actName2Kernel='relu', hidden2kernel=None, actName2GKN='tanh', dim2linear_trans=10, W2linear_trans=None,
              Wout=None, Bout=None, actName2out='linear'):
    """Construct edge feature for each point
        Args:
        point_set: float array -- (num_points, in_dim)
        coef_set: float array -- (num_points, 1)
        Ws2trans_input: (1, 1, in_dim, out_dim)
        Bs2trans_input:  (1, 1, 1, out_dim)
        actName2trans_input: string -- the name of activation function for changing the dimension of input-variable
        hiddens2trans_input: list or tuple
        scale_factor: float
        nn_idx: int array --(num_points, k)
        k_neighbors: int
        Ws2kernel:
        Bs2kernel:
        actName2Kernel: string
        hidden2kernel: list or tuple
        actName2GKN: the activation function of kernel for obtaining  the different neighbor point
        dim2linear_trans: (1, 1, out_dim, 1)
        W2linear_trans: (1, 1, k_neighbors, 1)

        Returns:
        new point_set: (num_points, out_dim)
    """
    if actName2GKN == 'relu':
        GNN_activation = tf.nn.relu
    elif actName2GKN == 'leaky_relu':
        GNN_activation = tf.nn.leaky_relu
    elif actName2GKN == 'srelu':
        GNN_activation = srelu
    elif actName2GKN == 's2relu':
        GNN_activation = s2relu
    elif actName2GKN == 'elu':
        GNN_activation = tf.nn.elu
    elif actName2GKN == 'sin':
        GNN_activation = mysin
    elif actName2GKN == 'tanh':
        GNN_activation = tf.nn.tanh
    elif actName2GKN == 'gauss':
        GNN_activation = gauss
    elif actName2GKN == 'mexican':
        GNN_activation = mexican
    elif actName2GKN == 'phi':
        GNN_activation = phi

    point_set_shape = point_set.get_shape()
    assert (len(point_set_shape)) == 2
    num_points = point_set_shape[0].value
    num_dims = point_set_shape[1].value

    # obtaining the coords of neighbors according to the corresponding index, then obtaining edge-feature
    select_idx = nn_idx                                           # indexes (num_points, k_neighbors)
    point_neighbors = tf.gather(point_set, select_idx)            # coords  (num_points, k_neighbors, dim2point)
    point_central = tf.expand_dims(point_set, axis=-2)            # (num_points, dim2point)-->(num_points, 1, dim2point)
    centroid_tilde = tf.tile(point_central, [1, k_neighbors, 1])  # (num_points, k_neighbors, dim2point)
    edges_feature = centroid_tilde - point_neighbors              # (num_points, k_neighbors, dim2point)

    # calculating the wight-coefficients for neighbors by edge-feature,then aggregating neighbors by wight-coefficients
    attention2neighbors = cal_attens2neighbors(edges_feature)     # (num_points, k_neighbors, dim2point)
    attention2neighbors = tf.nn.softmax(attention2neighbors)      # (num_points, 1, k_neighbors)

    # using a DNN model to change the dimension of input_data (point_set||coef_set) or (point_set)
    assert(len(hiddens2trans_input) >= 2)
    if len(coef_set) == 0:
        point_coef_set = point_set
    else:
        point_coef_set = tf.concat([point_set, coef_set], axis=-1)
    new_point_coef_set = DNN_FourierBase(point_coef_set, Ws2trans_input, Bs2trans_input, hiddens2trans_input,
                                         scale_factor, activate_name=actName2trans_input)

    # calculate the kernel by new_point_coef_set
    point_coef_neighbors = tf.gather(new_point_coef_set, select_idx)              # (num_points, k_neighbors, new_dim1)
    point_coef_central = tf.expand_dims(new_point_coef_set, axis=-2)            # (num_points, 1, new_dim1)
    point_coef_centroid_tile = tf.tile(point_coef_central, [1, k_neighbors, 1])   # (num_points, k_neighbors, new_dim1)
    # (num_points, k_neighbors, 2*new_dim1)
    central_neighbor2point_coef = tf.concat([point_coef_centroid_tile, point_coef_neighbors], axis=-1)
    # (num_points, k_neighbors, 2*new_dim1)-->(num_points, k_neighbors, new_dim1*new_dim2)
    kernel_neighbors = Kernel_DNN(central_neighbor2point_coef, Ws2kernel, Bs2kernel, hidden2kernel,
                                  activate_name=actName2Kernel)

    # (num_points, k_neighbors, new_dim1 * new_dim2)-->(num_points, k_neighbors, 1, new_dim1*new_dim2)
    kernel_neighbors = tf.expand_dims(kernel_neighbors, axis=-1)
    # (num_points, k_neighbors, 1, new_dim1*new_dim2) --.(num_points, k_neighbors, new_dim1, new_dim2)
    kernel_neighbors = tf.reshape(kernel_neighbors, shape=[-1, k_neighbors, dim2linear_trans, dim2linear_trans])
    # (num_points, k_neighbors, new_dim1) --> # (num_points, k_neighbors, 1, new_dim1)
    point_coef_neighbors = tf.expand_dims(point_coef_neighbors, axis=-2)
    kernel_matmul_neighbors = tf.matmul(point_coef_neighbors, kernel_neighbors)
    kernel_matmul_neighbors = tf.squeeze(kernel_matmul_neighbors, axis=-2)

    # aggregating neighbors by wight-coefficient
    atten_neighbors = tf.matmul(attention2neighbors, kernel_matmul_neighbors)
    squeeze2atten_neighbors = tf.squeeze(atten_neighbors)   # remove the dimension with 1 (num_points, new_dim1)

    # obtain the nwe point-set with new feature
    trans_new_point_coef_set = tf.matmul(new_point_coef_set, W2linear_trans)                   # (num_points, new_dim1)
    new_point_set = GNN_activation(tf.add(trans_new_point_coef_set, squeeze2atten_neighbors))  # (num_points, new_dim1)

    out_point_set = tf.add(tf.matmul(new_point_set, Wout), Bout)
    if actName2out == 'relu':
        out_point_set = tf.nn.relu(out_point_set)
    elif actName2out == 'leakly_relu':
        out_point_set = tf.nn.leaky_relu(out_point_set)
    elif actName2out == 'tanh':
        out_point_set = tf.nn.tanh(out_point_set)
    elif actName2out == 'elu':
        out_point_set = tf.nn.elu(out_point_set)
    elif actName2out == 'sigmoid':
        out_point_set = tf.nn.sigmoid(out_point_set)
    return out_point_set


# multiple layers GKN module
def HierarchicGKN(point_set, coef_set=None, PDE_type='Laplace', Ws2trans_input=None, Bs2trans_input=None,
                  actName2trans_input='tanh',
                  hiddens2trans_input=None, scale_factor=None, nn_idx=None, k_neighbors=10, Ws2kernel=None,
                  Bs2kernel=None, actName2Kernel='relu', hidden2kernel=None, actName2GKN='tanh', dim2linear_trans=10,
                  W2linear_trans=None, num2GKN=5, Wout=None, Bout=None, actName2out='linear'):
    """Construct edge feature for each point
        Args:
        point_set: float array -- (num_points, in_dim)
        coef_set: float array -- (num_points, 1)
        Ws2trans_input: (1, 1, in_dim, out_dim)
        Bs2trans_input:  (1, 1, 1, out_dim)
        actName2trans_input: string -- the name of activation function for changing the dimension of input-variable
        hiddens2trans_input: list or tuple
        scale_factor: float
        nn_idx: int array --(num_points, k)
        k_neighbors: int
        Ws2kernel:
        Bs2kernel:
        actName2Kernel: string
        hidden2kernel: list or tuple
        actName2GKN: the activation function of kernel for obtaining  the different neighbor point
        dim2linear_trans: (1, 1, out_dim, 1)
        W2linear_trans: (1, 1, k_neighbors, 1)

        Returns:
        new point_set: (num_points, out_dim)
    """
    if actName2GKN == 'relu':
        GNN_activation = tf.nn.relu
    elif actName2GKN == 'leaky_relu':
        GNN_activation = tf.nn.leaky_relu
    elif actName2GKN == 'srelu':
        GNN_activation = srelu
    elif actName2GKN == 's2relu':
        GNN_activation = s2relu
    elif actName2GKN == 'elu':
        GNN_activation = tf.nn.elu
    elif actName2GKN == 'sin':
        GNN_activation = mysin
    elif actName2GKN == 'tanh':
        GNN_activation = tf.nn.tanh
    elif actName2GKN == 'gauss':
        GNN_activation = gauss
    elif actName2GKN == 'mexican':
        GNN_activation = mexican
    elif actName2GKN == 'phi':
        GNN_activation = phi

    point_set_shape = point_set.get_shape()
    assert (len(point_set_shape)) == 2

    # obtaining the coords of neighbors according to the corresponding index, then obtaining edge-feature
    select_idx = nn_idx                                           # indexes (num_points, k_neighbors)
    point_neighbors = tf.gather(point_set, select_idx)            # coords  (num_points, k_neighbors, dim2point)
    point_central = tf.expand_dims(point_set, axis=-2)            # (num_points, dim2point)-->(num_points, 1, dim2point)
    centroid_tilde = tf.tile(point_central, [1, k_neighbors, 1])  # (num_points, k_neighbors, dim2point)
    edges_feature = centroid_tilde - point_neighbors              # (num_points, k_neighbors, dim2point)

    # calculating the wight-coefficients for neighbors by edge-feature,then aggregating neighbors by wight-coefficients
    attention2neighbors = cal_attens2neighbors(edges_feature)  # (num_points, k_neighbors, dim2point)
    attention2neighbors = tf.nn.softmax(attention2neighbors)   # (num_points, 1, k_neighbors)

    # using a DNN model to change the dimension of input_data (point_set||coef_set) or (point_set)
    assert (len(hiddens2trans_input) >= 2)
    if PDE_type == 'pLaplace_implicit' or PDE_type == 'pLaplace_explicit' or PDE_type == 'Possion_Boltzmann':
        point_coef_set = tf.concat([point_set, coef_set], axis=-1)
    else:
        point_coef_set = point_set
    new_point_coef_set = DNN_FourierBase(point_coef_set, Ws2trans_input, Bs2trans_input, hiddens2trans_input,
                                         scale_factor, activate_name=actName2trans_input)

    for i_layer in range(num2GKN):
        # W2linear_trans = Ws2linear_trans[i_layer]
        # calculate the kernel by new_point_coef_set
        point_coef_neighbors = tf.gather(new_point_coef_set, select_idx)             # (num_points, k_neighbors, new_dim1)
        point_coef_central = tf.expand_dims(new_point_coef_set, axis=-2)             # (num_points, 1, new_dim1)
        point_coef_centroid_tile = tf.tile(point_coef_central, [1, k_neighbors, 1])  # (num_points, k_neighbors, new_dim1)
        central_neighbor2point_coef = tf.concat([point_coef_centroid_tile, point_coef_neighbors], axis=-1)  # (num_points, k_neighbors, 2*new_dim1)
        # (num_points, k_neighbors, 2*new_dim1)-->(num_points, k_neighbors, new_dim1*new_dim2)
        kernel_neighbors = Kernel_DNN(central_neighbor2point_coef, Ws2kernel, Bs2kernel, hidden2kernel,
                                      activate_name=actName2Kernel)

        # (num_points, k_neighbors, new_dim1 * new_dim2)-->(num_points, k_neighbors, 1, new_dim1*new_dim2)
        kernel_neighbors = tf.expand_dims(kernel_neighbors, axis=-1)
        # (num_points, k_neighbors, 1, new_dim1*new_dim2) --.(num_points, k_neighbors, new_dim1, new_dim2)
        kernel_neighbors = tf.reshape(kernel_neighbors, shape=[-1, k_neighbors, dim2linear_trans, dim2linear_trans])
        # (num_points, k_neighbors, new_dim1) --> # (num_points, k_neighbors, 1, new_dim1)
        point_coef_neighbors = tf.expand_dims(point_coef_neighbors, axis=-2)
        kernel_matmul_neighbors = tf.matmul(point_coef_neighbors, kernel_neighbors)
        kernel_matmul_neighbors = tf.squeeze(kernel_matmul_neighbors, axis=-2)

        # aggregating neighbors by wight-coefficient
        atten_neighbors = tf.matmul(attention2neighbors, kernel_matmul_neighbors)
        squeeze2atten_neighbors = tf.squeeze(atten_neighbors)  # remove the dimension with 1 (num_points, new_dim1)

        # obtain the nwe point-set with new feature
        trans_new_point_coef_set = tf.matmul(new_point_coef_set, W2linear_trans)  # (num_points, new_dim1)
        new_point_coef_set = GNN_activation(
            tf.add(trans_new_point_coef_set, squeeze2atten_neighbors))            # (num_points, new_dim1)

    out_point_set = tf.add(tf.matmul(new_point_coef_set, Wout), Bout)
    if actName2out == 'relu':
        out_point_set = tf.nn.relu(out_point_set)
    elif actName2out == 'leakly_relu':
        out_point_set = tf.nn.leaky_relu(out_point_set)
    elif actName2out == 'tanh':
        out_point_set = tf.nn.tanh(out_point_set)
    elif actName2out == 'elu':
        out_point_set = tf.nn.elu(out_point_set)
    elif actName2out == 'sigmoid':
        out_point_set = tf.nn.sigmoid(out_point_set)
    return out_point_set


def test_SingleGNN_myConv():
    batchsize_it = 10
    init_dim = 2
    out_dim = 1
    out_dim2trans = 80
    knn2xy = 4
    freq = np.array([1])
    hiddenlist2trans_input = (5, 10, 20, 40)
    hiddenlist2kernel = (200, 250, 300, 350, 400)
    wlist2trans_input, blist2trans_input = Xavier_init_NN_Fourier(init_dim, out_dim2trans, hiddenlist2trans_input,
                                                                  Flag='WB2trans_input')
    wlist2kernel, blist2kernel = Xavier_init_NN(2*out_dim2trans, out_dim2trans*out_dim2trans, hiddenlist2kernel,
                                                Flag='WB2kernel')
    stddev_Wlinear = (2.0 / (out_dim2trans + out_dim2trans)) ** 0.5
    Wlinear_trans = tf.get_variable(name='Wlinear_trans', shape=(out_dim2trans, out_dim2trans),
                                    initializer=tf.random_normal_initializer(stddev=stddev_Wlinear), dtype=tf.float32)
    stddev_WBout = (2.0 / (out_dim2trans + out_dim)) ** 0.5
    Wout_trans = tf.get_variable(name='Wout_trans', shape=(out_dim2trans, out_dim),
                                 initializer=tf.random_normal_initializer(stddev=stddev_WBout), dtype=tf.float32)
    Bout_trans = tf.get_variable(name='Bout_trans', shape=(out_dim,),
                                 initializer=tf.random_normal_initializer(stddev=stddev_WBout), dtype=tf.float32)
    with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
        XY_mesh = tf.placeholder(tf.float32, name='X_it', shape=[None, init_dim])
        Ax=[]
        adj_matrix = pairwise_distance(XY_mesh)
        idx2knn = knn_excludeself(adj_matrix, k=knn2xy)
        U_hat = SingleGKN(XY_mesh, coef_set=Ax, Ws2trans_input=wlist2trans_input, Bs2trans_input=blist2trans_input,
                          actName2trans_input='tanh', hiddens2trans_input=hiddenlist2trans_input, scale_factor=freq,
                          nn_idx=idx2knn, k_neighbors=knn2xy, Ws2kernel=wlist2kernel, Bs2kernel=blist2kernel,
                          actName2Kernel='relu', hidden2kernel=hiddenlist2kernel, actName2GKN='tanh',
                          dim2linear_trans=out_dim2trans, W2linear_trans=Wlinear_trans, Wout=Wout_trans,
                          Bout=Bout_trans, actName2out='linear')

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        xy_mesh_batch = np.random.rand(batchsize_it, init_dim)

        u = sess.run(U_hat, feed_dict={XY_mesh: xy_mesh_batch})
        print(u)


def test_HierarchicGKN():
    batchsize_it = 10
    init_dim = 2
    out_dim = 1
    out_dim2trans = 80
    knn2xy = 4
    freq = np.array([1])
    hiddenlist2trans_input = (5, 10, 20, 40)
    hiddenlist2kernel = (200, 250, 300, 350, 400)
    wlist2trans_input, blist2trans_input = Xavier_init_NN_Fourier(init_dim, out_dim2trans, hiddenlist2trans_input,
                                                                  Flag='WB2trans_input')
    wlist2kernel, blist2kernel = Xavier_init_NN(2 * out_dim2trans, out_dim2trans * out_dim2trans, hiddenlist2kernel,
                                                Flag='WB2kernel')

    stddev_Wlinear = (2.0 / (out_dim2trans + out_dim2trans)) ** 0.5
    Wlinear_trans = tf.get_variable(name='Wlinear_trans', shape=(out_dim2trans, out_dim2trans),
                                    initializer=tf.random_normal_initializer(stddev=stddev_Wlinear), dtype=tf.float32)

    stddev_WBout = (2.0 / (out_dim2trans + out_dim)) ** 0.5
    Wout_trans = tf.get_variable(name='Wout_trans', shape=(out_dim2trans, out_dim),
                                 initializer=tf.random_normal_initializer(stddev=stddev_WBout), dtype=tf.float32)
    Bout_trans = tf.get_variable(name='Bout_trans', shape=(out_dim,),
                                 initializer=tf.random_normal_initializer(stddev=stddev_WBout), dtype=tf.float32)
    with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
        XY_mesh = tf.placeholder(tf.float32, name='X_it', shape=[None, init_dim])
        Ax = []
        adj_matrix = pairwise_distance(XY_mesh)
        idx2knn = knn_excludeself(adj_matrix, k=knn2xy)
        U_hat = HierarchicGKN(XY_mesh, coef_set=Ax, Ws2trans_input=wlist2trans_input, Bs2trans_input=blist2trans_input,
                              actName2trans_input='tanh', hiddens2trans_input=hiddenlist2trans_input, scale_factor=freq,
                              nn_idx=idx2knn, k_neighbors=knn2xy, Ws2kernel=wlist2kernel, Bs2kernel=blist2kernel,
                              actName2Kernel='relu', hidden2kernel=hiddenlist2kernel, actName2GKN='tanh',
                              dim2linear_trans=out_dim2trans, W2linear_trans=Wlinear_trans, num2GKN=3, Wout=Wout_trans,
                              Bout=Bout_trans, actName2out='linear')

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        xy_mesh_batch = np.random.rand(batchsize_it, init_dim)

        u = sess.run(U_hat, feed_dict={XY_mesh: xy_mesh_batch})
        print(u)


if __name__ == "__main__":
    # test_SingleGNN_myConv()
    test_HierarchicGKN()

