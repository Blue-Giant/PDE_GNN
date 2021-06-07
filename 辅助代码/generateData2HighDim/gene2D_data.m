clc;
clear all
close all
% dim = input('please input a integral number:');
dim = 2;
batch_size = 1600;
XY = rand(batch_size, dim);
save('testXY.mat', 'XY')