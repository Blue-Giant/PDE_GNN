clc;
clear all
close all
% dim = input('please input a integral number:');
dim = 3;
batch_size = 1600;
XYZ = rand(batch_size, dim);
save('testXYZ.mat', 'XYZ')