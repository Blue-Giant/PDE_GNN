A = cell(1,10);

for i = 1 : 10

    A(i) = {[(rand(1,1)-0),(rand(1,1)-0)]};

end

for i = 1 : 10

    centroid = cell2mat(A(i));

    rectangle('Position',[centroid(1)-0.5,centroid(2)-0.5,1,1],'Curvature',[1,1]);

    hold on;

    plot(centroid(1),centroid(2),'r+');

end

axis equal

axis([0 10 0 10]);

box on;