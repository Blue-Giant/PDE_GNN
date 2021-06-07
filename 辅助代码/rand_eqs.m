clc;
clear all
close all
x = (linspace(0,1,1000))';
K = 2;
xi1 = [-0.25,0.25];
xi2 = [-0.3,0.3];
a = 1.0;
alpha = 1.2;
sum = 0.0;
for k =1:K
    xi1k = xi1(k);
    xi2k = xi2(k);
%     temp = pow(k, -1.0*alpha)*(xi1k*sin(k*x)+xi2k*cos(k*x));
    temp = k^(-1.0*alpha).*(xi1k.*sin(k*x)+xi2k.*cos(k*x));
    sum = sum + temp;
end
a = a+0.5*sin(sum);
u = sum./a;
figure('name','u')
plot(x,u,'b-')
hold on

figure('name','gg')
gg = sin(0.2*sin(x)+0.2*cos(x));
plot(x,gg,'r-')
hold on