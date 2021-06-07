%x坐标范围
r1=unifrnd(3,5,100,1);
%y坐标范围
r2=unifrnd(3,4,100,1);
r=[r1,r2];
rr=[];
%写限制条件，我这是一个等腰三角形
for  i=1:100;    
    if  (r(i,1)<= 4&& r(i,2)<= r(i,1))||(r(i,1)>=4&& r(i,2)<=8- r(i,1))       
        rr=[rr;r(i,:)];    
    end
end
scatter(rr(:,1),rr(:,2),'k');

