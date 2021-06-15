"""
@author: LXA
 Date: 2021 年 6 月 5 日
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import time
import General_Laplace
import MS_LaplaceEqs
import MS_BoltzmannEqs
import MS_ConvectionEqs
import GNN_data
import matData2pLaplace
import matData2Laplace
import matData2Boltzmann
import matData2Convection
import GNN_base1
import GNN_tools
import GNN_PrintLog
import saveData
import plotData


def dictionary_out2file(R_dic, log_fileout):
    GNN_tools.log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    GNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)

    GNN_tools.log_string('Network model for transforming the input: %s\n' % str(R_dic['model2trans_input']), log_fileout)
    GNN_tools.log_string('hidden layer for transforming the input:%s\n' % str(R_dic['hiddens2input']), log_fileout)
    GNN_tools.log_string('The out-dim of transforming the input-data:%s\n' % str(R_dic['out_dim2trans']),log_fileout)

    GNN_tools.log_string('Network model for kernel: %s\n' % str(R_dic['model2kernel']), log_fileout)
    GNN_tools.log_string('hidden layer for kernel:%s\n' % str(R_dic['hiddens2kernel']), log_fileout)
    GNN_tools.log_string('Activate function for transforming the input: %s\n' % str(R_dic['actFunc2input']), log_fileout)
    GNN_tools.log_string('Activate function for graph kernel network: %s\n' % str(R_dic['actFunc2GKN']), log_fileout)
    GNN_tools.log_string('Activate function for train kernel: %s\n' % str(R_dic['actFunc2kernel']), log_fileout)
    GNN_tools.log_string('Activate function for output: %s\n' % str(R_dic['actFunc2out']), log_fileout)

    GNN_tools.log_string('Loss function: L2 loss\n', log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        GNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        GNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    GNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    GNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    if R_dic['activate_stop'] != 0:
        GNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        GNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)


def solve_multiScale_operator(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s%s.txt' % ('log2', 'train')
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    # 问题需要的设置
    batchsize2train = R['batchsize2train']
    batchsize2test = R['batchsize2test']

    wb_regular = R['regular_weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    hiddens2input = R['hiddens2input']
    hiddens2kernel = R['hiddens2kernel']
    actFunc2input = R['actFunc2input']
    actFunc2GKN = R['actFunc2GKN']
    actFunc2kernel = R['actFunc2kernel']

    input_dim = R['input_dim']
    out_dim = R['output_dim']
    out_dim2trans = R['out_dim2trans']
    iters2GKN = R['iteration_GKN']

    kneighbor1 = R['num2neighbor1']
    kneighbor2 = R['num2neighbor2']
    kneighbor3 = R['num2neighbor3']
    kneighbor4 = R['num2neighbor4']
    kneighbor5 = R['num2neighbor5']

    model2input = R['model2trans_input']
    model2kernel = R['model2kernel']
    mesh_number2train = R['mesh_number2train']
    mesh_number2test = R['mesh_number2test']
    p = 2
    epsilon = 0.1

    region_lb = 0.0
    region_rt = 1.0
    if R['PDE_type'] == 'general_Laplace':
        init_dim = input_dim
        # -laplace u = f
        region_lb = 0.0
        region_rt = 1.0
        f, u_true, u_left, u_right, u_bottom, u_top = General_Laplace.get_infos2Laplace_2D(
            input_dim=input_dim, out_dim=out_dim, left_bottom=region_lb, right_top=region_rt, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace_implicit':
        init_dim = input_dim + 1
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        if R['equa_name'] == 'multi_scale2D_5':
            region_lb = 0.0
            region_rt = 1.0
        else:
            region_lb = -1.0
            region_rt = 1.0
        u_true, f, A_eps, u_left, u_right, u_bottom, u_top = MS_LaplaceEqs.get_infos2pLaplace_2D(
            input_dim=input_dim, out_dim=out_dim, mesh_number=mesh_number2train, intervalL=0.0, intervalR=1.0,
            equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace_explicit':
        init_dim = input_dim + 1
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        if R['equa_name'] == 'multi_scale2D_7':
            region_lb = 0.0
            region_rt = 1.0
            u_true = MS_LaplaceEqs.true_solution2E7(input_dim, out_dim, eps=epsilon)
            u_left, u_right, u_bottom, u_top = MS_LaplaceEqs.boundary2E7(
                input_dim, out_dim, region_lb, region_rt, eps=epsilon)
            A_eps = MS_LaplaceEqs.elliptic_coef2E7(input_dim, out_dim, eps=epsilon)
    elif R['PDE_type'] == 'Possion_Boltzmann':
        init_dim = input_dim + 1
        # region_lb = -1.0
        region_lb = 0.0
        region_rt = 1.0
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        A_eps, kappa, u_true, u_left, u_right, u_top, u_bottom, f = MS_BoltzmannEqs.get_infos2Boltzmann_2D(
            equa_name=R['equa_name'], intervalL=region_lb, intervalR=region_rt)
    elif R['PDE_type'] == 'Convection_diffusion':
        init_dim = input_dim + 1
        region_lb = -1.0
        region_rt = 1.0
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']

        A_eps, Bx, By, u_true, u_left, u_right, u_top, u_bottom, f = MS_ConvectionEqs.get_infos2Convection_2D(
            equa_name=R['equa_name'], eps=epsilon, region_lb=0.0, region_rt=1.0)

    # 初始化权重和和偏置的模式
    flag2input = 'WB2trans_input'
    if R['model2trans_input'] == 'Linear_transform':
        stddev_WBin = (2.0 / (init_dim + out_dim2trans)) ** 0.5
        Ws2input = tf.get_variable(name='Win_trans', shape=(init_dim, out_dim2trans),
                                   initializer=tf.random_normal_initializer(stddev=stddev_WBin), dtype=tf.float32)
        Bs2input = tf.get_variable(name='Bin_trans', shape=(out_dim,),
                                   initializer=tf.random_normal_initializer(stddev=stddev_WBin), dtype=tf.float32)
    elif R['model2trans_input'] == 'Fourier_DNN':
        Ws2input, Bs2input = GNN_base1.Xavier_init_NN_Fourier(init_dim, out_dim2trans, hiddens2input, Flag=flag2input)
    else:
        Ws2input, Bs2input = GNN_base1.Xavier_init_NN(init_dim, out_dim2trans, hiddens2input, Flag=flag2input)

    flag2Kernel = 'WB2kernel'
    if R['model2kernel'] == 'DNN':
        Ws2kernel, Bs2kernel = GNN_base1.Xavier_init_NN(2 * out_dim2trans, out_dim2trans * out_dim2trans,
                                                        hiddens2kernel, Flag=flag2Kernel)
    elif R['model2kernel'] == 'Fourier_DNN':
        Ws2kernel, Bs2kernel = GNN_base1.Xavier_init_NN_Fourier(2 * out_dim2trans, out_dim2trans * out_dim2trans,
                                                                hiddens2kernel, Flag=flag2Kernel)

    Wslinear_trans = GNN_base1.Xavier_init_LinearTransW(in_size=out_dim2trans, out_size=out_dim2trans, num2GKN=5)

    stddev_WBout = (2.0 / (out_dim2trans + out_dim)) ** 0.5
    Wout = tf.get_variable(name='Wout', shape=(out_dim2trans, out_dim),
                           initializer=tf.random_normal_initializer(stddev=stddev_WBout), dtype=tf.float32)
    Bout = tf.get_variable(name='Bout', shape=(out_dim,),
                           initializer=tf.random_normal_initializer(stddev=stddev_WBout), dtype=tf.float32)
    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XY = tf.placeholder(tf.float32, name='XY', shape=[None, input_dim])
            Utrue = tf.placeholder(tf.float32, name='Utrue', shape=[None, out_dim])
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')

            # 最底层的邻居点多一些，最高层的邻居点最少
            adj_matrix = GNN_base1.pairwise_distance(XY)
            select_idx1 = GNN_base1.knn_excludeself(adj_matrix, k=kneighbor1)  # indexes (num_points, k_neighbors)
            select_idx2 = GNN_base1.knn_excludeself(adj_matrix, k=kneighbor2)
            select_idx3 = GNN_base1.knn_excludeself(adj_matrix, k=kneighbor3)
            select_idx4 = GNN_base1.knn_excludeself(adj_matrix, k=kneighbor4)
            select_idx5 = GNN_base1.knn_excludeself(adj_matrix, k=kneighbor5)

            # obtaining the coords of neighbors according to the corresponding index, then obtaining edge-feature
            point_neighbors1 = tf.gather(XY, select_idx1)    # coords  (num_points, k_neighbors, dim2point)
            point_central = tf.expand_dims(XY, axis=-2)      # (num_points, dim2point)-->(num_points, 1, dim2point)
            centroid_tile1 = tf.tile(point_central, [1, kneighbor1, 1])    # (num_points, k_neighbors, dim2point)
            edges_feature1 = centroid_tile1 - point_neighbors1             # (num_points, k_neighbors, dim2point)

            point_neighbors2 = tf.gather(XY, select_idx2)
            centroid_tile2 = tf.tile(point_central, [1, kneighbor2, 1])
            edges_feature2 = centroid_tile2 - point_neighbors2

            point_neighbors3 = tf.gather(XY, select_idx3)
            centroid_tile3 = tf.tile(point_central, [1, kneighbor3, 1])
            edges_feature3 = centroid_tile3 - point_neighbors3

            point_neighbors4 = tf.gather(XY, select_idx4)
            centroid_tile4 = tf.tile(point_central, [1, kneighbor4, 1])
            edges_feature4 = centroid_tile4 - point_neighbors4

            point_neighbors5 = tf.gather(XY, select_idx5)
            centroid_tile5 = tf.tile(point_central, [1, kneighbor5, 1])
            edges_feature5 = centroid_tile5 - point_neighbors5

            # calculating the wight-coefficients of neighbors by edge,then aggregating neighbors by wight-coefficients
            atten1_neighbors = GNN_base1.cal_attens2neighbors(edges_feature1)  # (num_points, k_neighbors, dim2point)
            atten1_neighbors = tf.nn.softmax(atten1_neighbors)                 # (num_points, 1, k_neighbors)

            atten2_neighbors = GNN_base1.cal_attens2neighbors(edges_feature2)
            atten2_neighbors = tf.nn.softmax(atten2_neighbors)

            atten3_neighbors = GNN_base1.cal_attens2neighbors(edges_feature3)
            atten3_neighbors = tf.nn.softmax(atten3_neighbors)

            atten4_neighbors = GNN_base1.cal_attens2neighbors(edges_feature4)
            atten4_neighbors = tf.nn.softmax(atten4_neighbors)

            atten5_neighbors = GNN_base1.cal_attens2neighbors(edges_feature5)
            atten5_neighbors = tf.nn.softmax(atten5_neighbors)

            freq_array = R['freqs']
            if R['PDE_type'] == 'pLaplace_implicit' or R['PDE_type'] == 'pLaplace_explicit' or \
                    R['PDE_type'] == 'Possion_Boltzmann':
                Xcoords = tf.reshape(XY[:, 0], shape=[-1, 1])
                Ycoords = tf.reshape(XY[:, 1], shape=[-1, 1])
                axy = A_eps(Xcoords, Ycoords)
            else:
                axy = []

            point_set = tf.concat([XY, axy], axis=-1)
            if model2input == 'Linear_transform':
                new_point_set = tf.add(tf.matmul(point_set, Ws2input), Bs2input)
            elif model2input == 'DNN':
                assert (len(hiddens2input) >= 2)
                new_point_set = GNN_base1.DNN(point_set, Ws2input, Bs2input, hiddens2input, activate_name=actFunc2input)
            else:
                assert (len(hiddens2input) >= 2)
                new_point_set = GNN_base1.DNN_FourierBase(
                    point_set, Ws2input, Bs2input, hiddens2input, freq_array, activate_name=actFunc2input)

            for i_layer in range(iters2GKN):
                W2trans = Wslinear_trans[i_layer]
                # calculate the kernel by new_point_coef_set
                new_point_central = tf.expand_dims(new_point_set, axis=-2)  # (num_points, 1, new_dim1)
                if i_layer == 0:
                    new_neighbors = tf.gather(new_point_set, select_idx1)                          # (num_points,k_neighbors,new_dim1)
                    new_point_central_tile = tf.tile(new_point_central, [1, kneighbor1, 1])        # (num_points,k_neighbors,new_dim1)
                    center_neighbor = tf.concat([new_point_central_tile, new_neighbors], axis=-1)  # (num_points, k_neighbors, 2*new_dim1)
                elif i_layer == 1:
                    new_neighbors = tf.gather(new_point_set, select_idx2)
                    new_point_central_tile = tf.tile(new_point_central, [1, kneighbor2, 1])
                    center_neighbor = tf.concat([new_point_central_tile, new_neighbors], axis=-1)
                elif i_layer == 2:
                    new_neighbors = tf.gather(new_point_set, select_idx3)
                    new_point_central_tile = tf.tile(new_point_central, [1, kneighbor3, 1])
                    center_neighbor = tf.concat([new_point_central_tile, new_neighbors], axis=-1)
                elif i_layer == 3:
                    new_neighbors = tf.gather(new_point_set, select_idx4)
                    new_point_central_tile = tf.tile(new_point_central, [1, kneighbor4, 1])
                    center_neighbor = tf.concat([new_point_central_tile, new_neighbors], axis=-1)
                elif i_layer == 4:
                    new_neighbors = tf.gather(new_point_set, select_idx5)
                    new_point_central_tile = tf.tile(new_point_central, [1, kneighbor5, 1])
                    center_neighbor = tf.concat([new_point_central_tile, new_neighbors], axis=-1)

                # (num_points, k_neighbors, 2*new_dim1)-->(num_points, k_neighbors, new_dim1*new_dim2)
                if model2kernel == 'DNN':
                    kernel_matrix = GNN_base1.Kernel_DNN(center_neighbor, Ws2kernel, Bs2kernel, hiddens2kernel,
                                                         activate_name=actFunc2kernel)
                else:
                    kernel_matrix = GNN_base1.DNN_FourierBase(center_neighbor, Ws2kernel, Bs2kernel, hiddens2kernel,
                                                              freq_array, activate_name=actFunc2kernel)

                # (num_points, k_neighbors, new_dim1*new_dim2) -->(num_points, k_neighbors, new_dim1, new_dim2)
                if i_layer == 0:
                    kernel_matrix = tf.reshape(kernel_matrix, shape=[-1, kneighbor1, out_dim2trans, out_dim2trans])
                    # (num_points, k_neighbors, new_dim1) --> # (num_points, k_neighbors, 1, new_dim1)
                    expand_new_neighbors = tf.expand_dims(new_neighbors, axis=-2)

                    kernel_matmul_neighbors = tf.matmul(expand_new_neighbors, kernel_matrix)
                    kernel_matmul_neighbors = tf.squeeze(kernel_matmul_neighbors, axis=-2)

                    # aggregating neighbors by wight-coefficient
                    atten_neighbors = tf.matmul(atten1_neighbors, kernel_matmul_neighbors)
                elif i_layer == 1:
                    kernel_matrix = tf.reshape(kernel_matrix, shape=[-1, kneighbor2, out_dim2trans, out_dim2trans])
                    expand_new_neighbors = tf.expand_dims(new_neighbors, axis=-2)
                    kernel_matmul_neighbors = tf.matmul(expand_new_neighbors, kernel_matrix)
                    kernel_matmul_neighbors = tf.squeeze(kernel_matmul_neighbors, axis=-2)
                    atten_neighbors = tf.matmul(atten2_neighbors, kernel_matmul_neighbors)
                elif i_layer == 2:
                    kernel_matrix = tf.reshape(kernel_matrix, shape=[-1, kneighbor3, out_dim2trans, out_dim2trans])
                    expand_new_neighbors = tf.expand_dims(new_neighbors, axis=-2)
                    kernel_matmul_neighbors = tf.matmul(expand_new_neighbors, kernel_matrix)
                    kernel_matmul_neighbors = tf.squeeze(kernel_matmul_neighbors, axis=-2)
                    atten_neighbors = tf.matmul(atten3_neighbors, kernel_matmul_neighbors)
                elif i_layer == 3:
                    kernel_matrix = tf.reshape(kernel_matrix, shape=[-1, kneighbor4, out_dim2trans, out_dim2trans])
                    expand_new_neighbors = tf.expand_dims(new_neighbors, axis=-2)
                    kernel_matmul_neighbors = tf.matmul(expand_new_neighbors, kernel_matrix)
                    kernel_matmul_neighbors = tf.squeeze(kernel_matmul_neighbors, axis=-2)
                    atten_neighbors = tf.matmul(atten4_neighbors, kernel_matmul_neighbors)
                elif i_layer == 4:
                    kernel_matrix = tf.reshape(kernel_matrix, shape=[-1, kneighbor5, out_dim2trans, out_dim2trans])
                    expand_new_neighbors = tf.expand_dims(new_neighbors, axis=-2)
                    kernel_matmul_neighbors = tf.matmul(expand_new_neighbors, kernel_matrix)
                    kernel_matmul_neighbors = tf.squeeze(kernel_matmul_neighbors, axis=-2)
                    atten_neighbors = tf.matmul(atten5_neighbors, kernel_matmul_neighbors)

                # remove the dimension with 1 (num_points, new_dim1)
                squeeze2atten_neighbors = tf.squeeze(atten_neighbors, axis=-2)

                # obtain the nwe point-set with new dimension
                trans_new_point_center = tf.matmul(new_point_set, W2trans)  # (num_points, new_dim)
                new_point_set = GNN_base1.activate_GKN(
                    tf.add(trans_new_point_center, squeeze2atten_neighbors))  # (num_points, new_dim)

            out_point_set = tf.add(tf.matmul(new_point_set, Wout), Bout)
            # UNN = tf.tanh(out_point_set)
            UNN = GNN_base1.activate_GKNout(out_point_set, actName2out='tanh')

            loss_u = 100*tf.reduce_mean(tf.square(UNN - Utrue))

            if R['regular_weight_model'] == 'L1':
                regular_WB2input = GNN_base1.regular_weights_biases_L1(Ws2input, Bs2input)
                regular_WB2kernnel = GNN_base1.regular_weights_biases_L1(Ws2kernel, Bs2kernel)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2input = GNN_base1.regular_weights_biases_L2(Ws2input, Bs2input)
                regular_WB2kernnel = GNN_base1.regular_weights_biases_L2(Ws2kernel, Bs2kernel)
            else:
                regular_WB2input = tf.constant(0.0)                                        # 无正则化权重参数
                regular_WB2kernnel = tf.constant(0.0)

            PWB2input = wb_regular * regular_WB2input
            PWB2kernel = wb_regular * regular_WB2kernnel
            PWB = PWB2input + PWB2kernel
            loss = loss_u + PWB                                  # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

            mse = tf.reduce_mean(tf.square(UNN - Utrue))
            rel = mse / tf.reduce_mean(tf.square(Utrue))

    t0 = time.time()
    loss_all, train_mse_all, train_rel_all = [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    if R['PDE_type'] == 'pLaplace_implicit' or R['PDE_type'] == 'pLaplace_explicit':
        test_xy = matData2pLaplace.get_meshData2pLaplace(equation_name=R['equa_name'], mesh_number=mesh_number2test)
        test_u = matData2pLaplace.get_soluData2pLaplace(equation_name=R['equa_name'], mesh_number=mesh_number2test)
    elif R['PDE_type'] == 'Possion_Boltzmann':
        if region_lb == (-1.0) and region_rt == 1.0:
            name2data_file = '11'
        else:
            name2data_file = '01'
        test_xy = matData2Boltzmann.get_meshData2Boltzmann(domain_lr=name2data_file, mesh_number=mesh_number2test)
    elif R['PDE_type'] == 'Convection_diffusion':
        if region_lb == (-1.0) and region_rt == 1.0:
            name2data_file = '11'
        else:
            name2data_file = '01'
        test_xy = matData2Boltzmann.get_meshData2Boltzmann(domain_lr=name2data_file, mesh_number=mesh_number2test)

    test_xy_bach, test_u_bach = GNN_data.randSample_existData2(coordData=test_xy, soluData=test_u, numSamples=batchsize2test)
    saveData.save_meshData2mat(test_xy_bach, dataName='meshXY', mesh_number=batchsize2test, outPath=R['FolderName'])
    saveData.save_meshData2mat(test_u_bach, dataName='meshU', mesh_number=batchsize2test, outPath=R['FolderName'])

    if R['PDE_type'] == 'pLaplace_implicit' or R['PDE_type'] == 'pLaplace_explicit':
        train_xy_bach = matData2pLaplace.get_meshData2pLaplace(equation_name=R['equa_name'], mesh_number=mesh_number2train)
        train_u_bach = matData2pLaplace.get_soluData2pLaplace(equation_name=R['equa_name'], mesh_number=mesh_number2train)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate
        for i_epoch in range(R['max_epoch'] + 1):
            if R['PDE_type'] == 'pLaplace_implicit' or R['PDE_type'] == 'pLaplace_explicit':
                xy_batch, u_batch = GNN_data.randSample_existData2(coordData=train_xy_bach, soluData=train_u_bach,
                                                                   numSamples=batchsize2train)
            tmp_lr = tmp_lr * (1 - lr_decay)

            _, loss_tmp, train_mse_tmp, train_rel_tmp, pwb = sess.run([train_my_loss, loss, mse, rel, PWB],
                feed_dict={XY: xy_batch, Utrue: u_batch, in_learning_rate: tmp_lr})

            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_rel_tmp)

            if i_epoch % 1000 == 0:
                test_epoch.append(i_epoch / 1000)
                run_times = time.time() - t0
                GNN_tools.print_and_log_train_one_epoch(i_epoch, run_times, tmp_lr, pwb, loss_tmp, train_mse_tmp,
                                                        train_rel_tmp, log_out=log_fileout)
                u_nn2test, test_mse_tmp, test_rel_tmp = sess.run([UNN, mse, rel],
                                                                 feed_dict={XY: test_xy_bach, Utrue: test_u_bach})
                test_mse_all.append(test_mse_tmp)
                test_rel_all.append(test_rel_tmp)
                GNN_tools.print_and_log_test_one_epoch(test_mse_tmp, test_rel_tmp, log_out=log_fileout)

        # ------------------- save the testing results into mat file and plot them -------------------------
        saveData.save_trainLoss2mat(loss, lossName='loss', outPath=R['FolderName'])
        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=actFunc2GKN, outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=actFunc2GKN, outPath=R['FolderName'])
        plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=actFunc2GKN, seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)

        # ----------------------  save testing results to mat files, then plot them --------------------------------
        saveData.save_2testSolus2mat(u_nn2test, test_u_bach, actName='utrue', actName1=actFunc2GKN,
                                     outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=actFunc2GKN, outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=actFunc2GKN,
                                  seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)


if __name__ == "__main__":
    R={}
    # -------------------------------------- CPU or GPU 选择 -----------------------------------------------
    R['gpuNo'] = 0
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

        if tf.test.is_gpu_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 文件保存路径设置
    # store_file = 'Laplace2D'
    store_file = 'pLaplace2D'
    # store_file = 'Boltzmann2D'
    # store_file = 'Convection2D'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    R['input_dim'] = 2                # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1               # 输出维数

    # ---------------------------- Setup of multi-scale problem-------------------------------
    if store_file == 'Laplace2D':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace2D':
        R['PDE_type'] = 'pLaplace_implicit'
        # R['equa_name'] = 'multi_scale2D_1'
        # R['equa_name'] = 'multi_scale2D_2'
        R['equa_name'] = 'multi_scale2D_3'
        # R['equa_name'] = 'multi_scale2D_4'

        # R['laplace_opt'] = 'pLaplace_explicit'
        # R['equa_name'] = 'multi_scale2D_6'
        # R['equa_name'] = 'multi_scale2D_7'

    if R['PDE_type'] == 'general_Laplace':
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
    elif R['PDE_type'] == 'pLaplace_implicit':
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
    elif R['PDE_type'] == 'pLaplace_explicit':
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2

    # R['getData_model2train'] = 'random_generate'
    R['getData_model2train'] = 'load_data'

    # R['getData_model2test'] = 'random_generate'
    R['getData_model2test'] = 'load_data'

    R['mesh_number2train'] = 5
    R['mesh_number2test'] = 6
    # R['batchsize2train'] = 500
    # R['batchsize2test'] = 400
    R['batchsize2train'] = 400
    R['batchsize2test'] = 200

    # ---------------------------- Setup of DNN -------------------------------
    R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    # R['regular_weight_model'] = 'L2'
    R['regular_weight_biases'] = 0.000                  # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0025                 # Regularization parameter for weights

    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['optimizer_name'] = 'Adam'                          # 优化器

    # R['model2trans_input'] = 'Linear_transform'           # 使用的网络模型
    # R['model2trans_input'] = 'DNN'
    R['model2trans_input'] = 'Fourier_DNN'

    R['model2kernel'] = 'DNN'
    # R['model2kernel'] = 'Fourier_DNN'

    if R['model2trans_input'] == 'DNN':
        # 3+ 3*10 + 10*10+ 10*20 + 20*40 = 1133
        R['hiddens2input'] = (10, 10, 20, 20)
        R['out_dim2trans'] = 30
    elif R['model2trans_input'] == 'Fourier_DNN':
        # 3+ 3*5 + 10*10+ 10*20 + 20*40 = 1118
        R['hiddens2input'] = (5, 10, 20, 20)
        R['out_dim2trans'] = 30
    else:
        # 3*100 = 300
        R['hiddens2input'] = (0)
        R['out_dim2trans'] = 100
    if R['model2kernel'] == 'DNN':
        # R['hiddens2kernel'] = (200, 200, 300, 300)
        R['hiddens2kernel'] = (200, 200, 300, 400)
    else:
        # R['hiddens2kernel'] = (100, 200, 300, 300)
        R['hiddens2kernel'] = (100, 200, 300, 400)

    # R['actFunc2GKN'] = 'relu'
    R['actFunc2GKN'] = 'tanh'
    # R['actFunc2GKN']' = leaky_relu'
    # R['actFunc2GKN'] = 'srelu'
    # R['actFunc2GKN'] = 's2relu'

    R['actFunc2input'] = 'tanh'
    # R['actFunc2input'] = 'relu'

    R['actFunc2kernel'] = 'relu'
    # R['actFunc2kernel'] = 'tanh'

    R['actFunc2out'] = 'linear'

    R['freqs'] = np.array([1])
    # R['freqs'] = np.array([1, 1, 2, 3, 4])

    # R['iteration_GKN'] = 5
    # R['num2neighbor1'] = 30
    # R['num2neighbor2'] = 20
    # R['num2neighbor3'] = 15
    # R['num2neighbor4'] = 10
    # R['num2neighbor5'] = 5
    #
    # R['iteration_GKN'] = 4
    # R['num2neighbor1'] = 30
    # R['num2neighbor2'] = 20
    # R['num2neighbor3'] = 10
    # R['num2neighbor4'] = 5
    # R['num2neighbor5'] = 5

    R['iteration_GKN'] = 3
    R['num2neighbor1'] = 25
    R['num2neighbor2'] = 18
    R['num2neighbor3'] = 10
    R['num2neighbor4'] = 5
    R['num2neighbor5'] = 5

    solve_multiScale_operator(R)

