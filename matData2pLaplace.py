# !python3
# -*- coding: utf-8 -*-
# author: flag

import numpy as np
import scipy.io


# load the data from matlab of .mat
def loadMatlabIdata(filename=None):
    data = scipy.io.loadmat(filename, mat_dtype=True, struct_as_record=True)  # variable_names='CATC'
    return data


def get_meshData2pLaplace(equation_name=None, mesh_number=2):
    if equation_name == 'multi_scale2D_1':
        test_meshXY_file = 'dataMat2pLaplace/E1/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_2':
        test_meshXY_file = 'dataMat2pLaplace/E2/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_3':
        test_meshXY_file = 'dataMat2pLaplace/E3/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_4':
        test_meshXY_file = 'dataMat2pLaplace/E4/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_5':
        test_meshXY_file = 'dataMat2pLaplace/E5/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_6':
        test_meshXY_file = 'dataMat2pLaplace/E6/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_7':
        assert(mesh_number == 6)
        test_meshXY_file = 'dataMat2pLaplace/E7/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_XY = loadMatlabIdata(test_meshXY_file)
    XY = mesh_XY['meshXY']
    test_mesh_data = np.transpose(XY, (1, 0))
    return test_mesh_data


def get_soluData2pLaplace(equation_name=None, mesh_number=2):
    if equation_name == 'multi_scale2D_1':
        test_meshXY_file = 'dataMat2pLaplace/E1/' + str('u_true') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_2':
        test_meshXY_file = 'dataMat2pLaplace/E2/' + str('u_true') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_3':
        test_meshXY_file = 'dataMat2pLaplace/E3/' + str('u_true') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_4':
        test_meshXY_file = 'dataMat2pLaplace/E4/' + str('u_true') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_5':
        test_meshXY_file = 'dataMat2pLaplace/E5/' + str('u_true') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_6':
        assert (mesh_number == 6)
        test_meshXY_file = 'dataMat2pLaplace/E6/' + str('u_true') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_7':
        test_meshXY_file = 'dataMat2pLaplace/E7/' + str('u_true') + str(mesh_number) + str('.mat')
    mesh_U = loadMatlabIdata(test_meshXY_file)
    Udata = mesh_U['u_true']
    return Udata


if __name__ == '__main__':
    equaName = 'multi_scale2D_3'
    resolution = 3
    meshXY = get_meshData2pLaplace(equation_name=equaName, mesh_number=resolution)
    print(meshXY)
    meshU = get_soluData2pLaplace(equation_name=equaName, mesh_number=resolution)
    print(meshU)