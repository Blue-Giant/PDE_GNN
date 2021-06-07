% rectangle('position',[1,1,5,5],'curvature',[1,1],'edgecolor','r','facecolor','g');
% 'position',[1,1,5,5]��ʾ�ӣ�1,1���㿪ʼ��Ϊ5����Ϊ5��
% 'curvature',[1,1]��ʾx,y�����ϵ����ʶ�Ϊ1������Բ����
% 'edgecolor','r'��ʾ�߿���ɫ�Ǻ�ɫ��
% 'facecolor','g'��ʾ���������ɫΪ��ɫ��

clear all
clc
close all

rectangle('position',[0-1,-1,2,2],'edgecolor','k', 'facecolor','y');

A = cell(1,5);
A(1) = {[0.3, 0.3]};
A(2) = {[-0.6, 0.5]};
A(3) = {[-0.7, -0.5]};
A(4) = {[-0.2, -0.5]};
A(5) = {[0.4, -0.7]};

radius=zeros(1,20);
radius(1)=0.4;
radius(2)=0.3;
radius(3)=0.3;
radius(4)=0.225;
radius(5)=0.3;



for i = 1 : 5
    centroid = cell2mat(A(i));
    rectangle('Position',[centroid(1),centroid(2),radius(i),radius(i)],'Curvature',[1,1], 'facecolor','w');
    hold on;
end