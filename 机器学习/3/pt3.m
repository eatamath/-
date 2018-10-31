
% 最小二乘法配图
sz=10;
dim=2;
sc=10;
bias=10;
x=[];
y=[];
z=[];


xx=linspace(-0.5*sc,0.5*sc,100);
yy=linspace(-0.5*sc,0.5*sc,100);
[X,Y]=meshgrid(xx,yy);
Z=X+Y;
figure();
mesh(X,Y,Z);
hold on;
xlabel('x1');
ylabel('x2');
zlabel('z');
for i =1:1:sz*sz*0.5
    a=floor(rand()*100+1);
    b=floor(rand()*100+1);
    scatter3(xx(a),yy(b),rand()*bias-0.5*bias+xx(a)+yy(b),'red','filled');

end


