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
import GNN_base
import GNN_tools
import GNN_PrintLog
import saveData
import plotData


def solve_multiScale_operator(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s%s.txt' % ('log2', R['activate_func'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    GNN_PrintLog.dictionary_out2file(R, log_fileout)

    # 一般 laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    wb_regular = R['regular_weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    hiddens_list = R['hidden_layers']
    act_func = R['activate_func']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    h2kernel = R['height2kernel']
    w2kernel = R['width2kernel']
    knn2xy = R['num2neighbor']

    # p laplace 问题需要的额外设置, 先预设一下
    p = 2
    epsilon = 0.1
    mesh_number = 2

    region_lb = 0.0
    region_rt = 1.0
    if R['PDE_type'] == 'general_Laplace':
        dim2kernel = input_dim
        # -laplace u = f
        region_lb = 0.0
        region_rt = 1.0
        f, u_true, u_left, u_right, u_bottom, u_top = General_Laplace.get_infos2Laplace_2D(
            input_dim=input_dim, out_dim=out_dim, left_bottom=region_lb, right_top=region_rt, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace_implicit':
        dim2kernel = input_dim + 1
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        if R['equa_name'] == 'multi_scale2D_5':
            region_lb = 0.0
            region_rt = 1.0
        else:
            region_lb = -1.0
            region_rt = 1.0
        u_true, f, A_eps, u_left, u_right, u_bottom, u_top = MS_LaplaceEqs.get_infos2pLaplace_2D(
            input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], intervalL=0.0, intervalR=1.0,
            equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace_explicit':
        dim2kernel = input_dim + 1
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        if R['equa_name'] == 'multi_scale2D_7':
            region_lb = 0.0
            region_rt = 1.0
            u_true = MS_LaplaceEqs.true_solution2E7(input_dim, out_dim, eps=epsilon)
            u_left, u_right, u_bottom, u_top = MS_LaplaceEqs.boundary2E7(
                input_dim, out_dim, region_lb, region_rt, eps=epsilon)
            A_eps = MS_LaplaceEqs.elliptic_coef2E7(input_dim, out_dim, eps=epsilon)
    elif R['PDE_type'] == 'Possion_Boltzmann':
        dim2kernel = input_dim + 1
        # region_lb = -1.0
        region_lb = 0.0
        region_rt = 1.0
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        A_eps, kappa, u_true, u_left, u_right, u_top, u_bottom, f = MS_BoltzmannEqs.get_infos2Boltzmann_2D(
            equa_name=R['equa_name'], intervalL=region_lb, intervalR=region_rt)
    elif R['PDE_type'] == 'Convection_diffusion':
        dim2kernel = input_dim + 1
        region_lb = -1.0
        region_rt = 1.0
        p_index = R['order2pLaplace_operator']
        epsilon = R['epsilon']
        mesh_number = R['mesh_number']
        A_eps, Bx, By, u_true, u_left, u_right, u_top, u_bottom, f = MS_ConvectionEqs.get_infos2Convection_2D(
            equa_name=R['equa_name'], eps=epsilon, region_lb=0.0, region_rt=1.0)

    # 初始化权重和和偏置的模式
    flag2kernel = 'Kernel2WB'
    if R['model'] == 'PDE_GNN_Fourier':
        W2NN, B2NN = GNN_base.Xavier_init_Fourier_Kernel(height2kernel=h2kernel, width2kernel=w2kernel,
                                                         in_size=dim2kernel, out_size=out_dim,
                                                         hidden_layers=hiddens_list, Flag=flag2kernel)
    else:
        W2NN, B2NN = GNN_base.Xavier_init_Kernel(height2kernel=h2kernel, width2kernel=w2kernel, in_size=dim2kernel,
                                                 out_size=out_dim, hidden_layers=hiddens_list, Flag=flag2kernel)

    flag2attenKernel = 'Kernel2atten'
    W2atten, B2atten = GNN_base.Xavier_init_attenKernel(height2kernel=h2kernel, width2kernel=w2kernel,
                                                        in_size=dim2kernel, out_size=out_dim, kneigh=knn2xy,
                                                        hidden_layers=hiddens_list, Flag=flag2attenKernel)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XY_train = tf.placeholder(tf.float32, name='XY_train', shape=[batchsize_it, input_dim])  # * 行 2 列
            XY_test = tf.placeholder(tf.float32, name='XY_test', shape=[batchsize_it, input_dim])  # * 行 2 列
            Utrue = tf.placeholder(tf.float32, name='Utrue', shape=[batchsize_bd, out_dim])        # * 行 2 列
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')

            freq_array = R['freqs']
            adj_matrix2train = GNN_base.pairwise_distance(XY_train)
            idx2knn2train = GNN_base.knn_excludeself(adj_matrix2train, k=knn2xy)
            if R['PDE_type'] == 'pLaplace_implicit' or R['PDE_type'] == 'pLaplace_explicit' or \
                R['PDE_type'] == 'Possion_Boltzmann':
                X_it = tf.reshape(XY_train[:, 0], shape=[-1, 1])
                Y_it = tf.reshape(XY_train[:, 1], shape=[-1, 1])
                axy = A_eps(X_it, Y_it)
                XY = tf.concat([XY_train, axy], axis=-1)
            else:
                XY = XY_train

            if R['model'] == 'PDE_GNN':
                UNN2train = GNN_base.HierarchicGNN_myConv(
                    XY, Weight_list=W2NN, Bias_list=B2NN, nn_idx=idx2knn2train, k_neighbors=knn2xy,
                    activate_name=act_func, scale_trans=R['scale_trans'], freqs=freq_array,
                    opt2cal_atten=R['opt2calc_atten'], actName2atten='relu', kernels2atten=W2atten, biases2atten=B2atten,
                    hiddens=hiddens_list)
            else:
                UNN2train = GNN_base.FourierHierarchicGNN_myConv(
                    XY, Weight_list=W2NN, Bias_list=B2NN, nn_idx=idx2knn2train, k_neighbors=knn2xy,
                    activate_name=act_func, scale_trans=R['scale_trans'], freqs=freq_array,
                    opt2cal_atten=R['opt2calc_atten'], actName2atten='relu', kernels2atten=W2atten, biases2atten=B2atten,
                    hiddens=hiddens_list)
            adj_matrix2test = GNN_base.pairwise_distance(XY_test)
            idx2knn2test = GNN_base.knn_excludeself(adj_matrix2test, k=knn2xy)
            if R['PDE_type'] == 'pLaplace_implicit' or R['PDE_type'] == 'pLaplace_explicit' or \
                    R['PDE_type'] == 'Possion_Boltzmann':
                X_it = tf.reshape(XY_test[:, 0], shape=[-1, 1])
                Y_it = tf.reshape(XY_test[:, 1], shape=[-1, 1])
                axy = A_eps(X_it, Y_it)
                XY = tf.concat([XY_test, axy], axis=-1)
            else:
                XY = XY_test

            if R['model'] == 'PDE_GNN':
                UNN2test = GNN_base.HierarchicGNN_myConv(
                    XY, Weight_list=W2NN, Bias_list=B2NN, nn_idx=idx2knn2test, k_neighbors=knn2xy,
                    activate_name=act_func, scale_trans=R['scale_trans'], freqs=freq_array,
                    opt2cal_atten=R['opt2calc_atten'], actName2atten='relu', kernels2atten=W2atten,
                    biases2atten=B2atten, hiddens=hiddens_list)
            else:
                UNN2test = GNN_base.FourierHierarchicGNN_myConv(
                    XY, Weight_list=W2NN, Bias_list=B2NN, nn_idx=idx2knn2test, k_neighbors=knn2xy,
                    activate_name=act_func, scale_trans=R['scale_trans'], freqs=freq_array,
                    opt2cal_atten=R['opt2calc_atten'], actName2atten='relu', kernels2atten=W2atten,
                    biases2atten=B2atten, hiddens=hiddens_list)
            if R['PDE_type'] != 'pLaplace_implicit':
                Utrue = u_true(X_it, Y_it)

            loss_u = tf.reduce_mean(tf.square(UNN2train - Utrue))

            if R['regular_weight_model'] == 'L1':
                regular_WB = GNN_base.regular_weights_biases_L1(W2NN, B2NN)          # 正则化权重和偏置 L1正则化
            elif R['regular_weight_model'] == 'L2':
                regular_WB = GNN_base.regular_weights_biases_L2(W2NN, B2NN)          # 正则化权重和偏置 L2正则化
            else:
                regular_WB = tf.constant(0.0)                                        # 无正则化权重参数

            PWB = wb_regular * regular_WB
            loss = loss_u + PWB                                                      # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

            train_mse = loss_u
            train_rel = train_mse / tf.reduce_mean(tf.square(Utrue))

    t0 = time.time()
    loss_all, train_mse_all, train_rel_all = [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        test_bach_size = 1600
        size2test = 40
        # test_bach_size = 4900
        # size2test = 70
        # test_bach_size = 10000
        # size2test = 100
        test_xy_bach = GNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
        saveData.save_testData_or_solus2mat(test_xy_bach, dataName='testXY', outPath=R['FolderName'])
    else:
        if R['PDE_type'] == 'pLaplace_implicit' or R['PDE_type'] == 'pLaplace_explicit':
            test_xy_bach = matData2pLaplace.get_data2pLaplace(equation_name=R['equa_name'], mesh_number=mesh_number)
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))
            saveData.save_meshData2mat(test_xy_bach, dataName='meshXY', mesh_number=mesh_number, outPath=R['FolderName'])
        elif R['PDE_type'] == 'Possion_Boltzmann':
            if region_lb == (-1.0) and region_rt == 1.0:
                name2data_file = '11'
            else:
                name2data_file = '01'
            test_xy_bach = matData2Boltzmann.get_meshData2Boltzmann(domain_lr=name2data_file, mesh_number=mesh_number)
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))
            saveData.save_meshData2mat(test_xy_bach, dataName='meshXY', mesh_number=mesh_number,
                                       outPath=R['FolderName'])
        elif R['PDE_type'] == 'Convection_diffusion':
            if region_lb == (-1.0) and region_rt == 1.0:
                name2data_file = '11'
            else:
                name2data_file = '01'
            test_xy_bach = matData2Boltzmann.get_meshData2Boltzmann(domain_lr=name2data_file, mesh_number=mesh_number)
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))
            saveData.save_meshData2mat(test_xy_bach, dataName='meshXY', mesh_number=mesh_number,
                                       outPath=R['FolderName'])
        else:
            test_xy_bach = matData2Laplace.get_randData2Laplace(dim=input_dim, data_path='dataMat_highDim')
            size2batch = np.shape(test_xy_bach)[0]
            size2test = int(np.sqrt(size2batch))

    if R['PDE_type'] == 'pLaplace_implicit':
        mesh2train = 5
        train_xy_bach = matData2pLaplace.get_meshData2pLaplace(equation_name=R['equa_name'], mesh_number=mesh2train)
        train_u_bach = matData2pLaplace.get_soluData2pLaplace(equation_name=R['equa_name'], mesh_number=mesh2train)
    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        train_option = True
        for i_epoch in range(R['max_epoch'] + 1):
            if R['PDE_type'] == 'pLaplace_implicit':
                xy_batch, u_batch = GNN_data.randSample_existData2(coordData=train_xy_bach, soluData=train_u_bach,
                                                                   numSamples=100)
            else:
                xy_batch = GNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            tmp_lr = tmp_lr * (1 - lr_decay)

            _, loss_tmp, train_mse_tmp, train_res_tmp, pwb = sess.run(
                [train_my_loss, loss, train_mse, train_rel, PWB],
                feed_dict={XY_train: xy_batch, Utrue: u_batch, in_learning_rate: tmp_lr})

            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_res_tmp)

            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                GNN_tools.print_and_log_train_one_epoch(i_epoch, run_times, tmp_lr, pwb, loss_tmp, train_mse_tmp,
                                                        train_res_tmp, log_out=log_fileout)
                u_nn2test = sess.run(UNN2test,  feed_dict={XY_test: test_xy_bach})

        # ------------------- save the testing results into mat file and plot them -------------------------
        saveData.save_trainLoss2mat(loss, lossName='loss', outPath=R['FolderName'])
        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act_func, seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)


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
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
        # R['batch_size2interior'] = 100  # 内部训练数据的批大小
        # R['batch_size2boundary'] = 25  # 边界训练数据的批大小
        R['batch_size2interior'] = 1000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 500  # 边界训练数据的批大小
    elif R['PDE_type'] == 'pLaplace_implicit':
        # 频率设置
        mesh_number = input('please input mesh_number =')  # 由终端输入的会记录为字符串形式
        R['mesh_number'] = int(mesh_number)  # 字符串转为浮点

        # 频率设置
        epsilon = input('please input epsilon =')  # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)  # 字符串转为浮点

        # 问题幂次
        order2p_laplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2p_laplace)
        R['order2pLaplace_operator'] = order

        R['batch_size2interior'] = 1000  # 内部训练数据的批大小
        if R['mesh_number'] == 2:
            R['batch_size2boundary'] = 25  # 边界训练数据的批大小
        elif R['mesh_number'] == 3:
            R['batch_size2boundary'] = 50  # 边界训练数据的批大小
        elif R['mesh_number'] == 4:
            R['batch_size2boundary'] = 100  # 边界训练数据的批大小
        elif R['mesh_number'] == 5:
            R['batch_size2boundary'] = 200  # 边界训练数据的批大小
        elif R['mesh_number'] == 6:
            R['batch_size2boundary'] = 400  # 边界训练数据的批大小
    elif R['laplace_opt'] == 'pLaplace_explicit':
        # 频率设置
        epsilon = input('please input epsilon =')  # 由终端输入的会记录为字符串形式
        R['epsilon'] = float(epsilon)  # 字符串转为浮点

        # 问题幂次
        order2p_laplace = input('please input the order(a int number) to p-laplace:')
        order = float(order2p_laplace)
        R['order2pLaplace_operator'] = order
        R['batch_size2interior'] = 1000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 500  # 边界训练数据的批大小

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

    R['hidden_layers'] = (5, 10, 20, 40, 20, 5)
    # R['hidden_layers'] = (100, 80, 80, 60, 40, 40)
    # R['hidden_layers'] = (200, 100, 80, 50, 30)

    R['model'] = 'PDE_GNN'                               # 使用的网络模型
    # R['model'] = 'PDE_GNN_Fourier'                       # 使用的网络模型

    # R['activate_func'] = 'relu'
    R['activate_func'] = 'tanh'
    # R['activate_func']' = leaky_relu'
    # R['activate_func'] = 'srelu'
    # R['activate_func'] = 's2relu'
    # R['activate_func'] = 'gauss'
    # R['activate_func'] = 'elu'
    # R['activate_func'] = 'phi'

    R['freqs'] = np.concatenate(([1], np.arange(1, 50 - 1)), axis=0)
    R['height2kernel'] = 1
    R['width2kernel'] = 1
    R['num2neighbor'] = 20
    R['scale_trans'] = True
    if R['PDE_type'] == 'general_Laplace':
        R['scale_trans'] = False

    R['opt2calculate_attention'] = 'dist_attention'
    R['opt2calculate_attention'] = 'my_cal_attention'

    solve_multiScale_operator(R)

