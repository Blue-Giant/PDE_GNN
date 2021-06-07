clc;
clear all
close all
% dim = input('please input a integral number:');
dim = 4;
batch_size = 1600;
XYZS = rand(batch_size, dim);
save('testXYZS.mat', 'XYZS')