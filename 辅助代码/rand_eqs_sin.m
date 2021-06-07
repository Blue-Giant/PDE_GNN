clc;
clear all
close all
x = (linspace(0,1,1000))';
K = 2;
xi1 = 0.25;
xi2 = -0.25;
u= sin(xi1*sin(pi*x)+0.5*xi2*sin(2*pi*x));
ux = cos(xi1*sin(pi*x)+0.5*xi2*sin(2*pi*x)).*pi.*(xi1*cos(pi*x)+xi2*cos(2*pi*x));
uxx = sin(xi1*sin(pi*x)+0.5*xi2*sin(2*pi*x)).*pi.*(xi1*cos(pi*x)+xi2*cos(2*pi*x)).*pi.*(xi1*cos(pi*x)+xi2*cos(2*pi*x))...
    -cos(xi1*sin(pi*x)+0.5*xi2*sin(2*pi*x)).*pi.*(-pi*xi1*sin(pi*x)-2*pi*xi2*sin(2*pi*x));
figure('name','u')
plot(x,u,'b-')
hold on

figure('name','du')
plot(x,ux,'r-')
hold on

figure('name','ddu')
plot(x,uxx,'m-')
hold on
