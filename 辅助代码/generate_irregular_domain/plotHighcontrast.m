clc
clear all
close all
alpha_0=1;
alpha_1 = 1000;
r0=0.5;
N=200;
x = linspace(-1,1,N);
y = linspace(-1,1,N);
[meshX,meshY] = meshgrid(x,y);
A = zeros(N,N);

for ix=1:N
    for iy=1:N
        if x(ix)*x(ix)+y(iy)*y(iy)<r0
           A(ix,iy)= alpha_1;
        else
            A(ix,iy)= alpha_0;
        end
    end
end

figure(1)
surf(meshX,meshY,A)
colorbar('ytick',[-0,1000])
shading interp;             %过渡均匀化，去掉网格线

% % c=zeros(size(x)); %获得0阵大小和xA相同
% 
% % for ix=1:N
% %     for iy=1:N
% %         if x(ix)*x(ix)+y(iy)*y(iy)<r0
% %            c(ix,iy,1)= 10;
% %            c(ix,iy,2)= 0;
% %            c(ix,iy,3)= 0;
% %         else
% %            c(ix,iy,1)= 0;
% %            c(ix,iy,2)= 0;
% %            c(ix,iy,3)= 1;
% %         end
% %     end
% % end
% 
% surf(meshX,meshY,A)
% mycolor6 = [
% 0 0 1
% 0 1 0
% 1 0 0];
% 
% colormap(mycolor6)
% % shading interp;%过渡均匀化，去掉网格线
% grid on
% % colormap (flipud(jet))%把最值对应的颜色互换
% colormap (flipud(jet(37)))%使用jet颜色地图中的37种颜色
% % colormap hot