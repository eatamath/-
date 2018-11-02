# -*- coding: utf-8 -*-
"""
Created on Wed Aug 08 18:50:18 2018
Linear Rregression
@author: lenovo
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pywin as win32api
import pandas as pd
import csv
import cx_Oracle as oc
import sqlite3 as sql
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D as aa
from sklearn import linear_model as limd
from sklearn.linear_model import LinearRegression as LR
from sklearn import preprocessing as pp

from sklearn.datasets import load_boston as lb
import sklearn.cross_validation as cv
from sklearn.preprocessing import StandardScaler as ss
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


'''
实例 3.1 随机曲面拟合 z=x1+x2^2+x1*x2 最小二乘法
'''
'''
print "随机曲面拟合"
from sklearn.model_selection import train_test_split
# 模型数据处理
scale=2 # 坐标数据尺度
# 随机产生二维数据
x1=np.random.rand(100,1)*scale 
x2=np.random.rand(100,1)*scale
X=np.ones([100,3]) # 模型的输入数据
# 矩阵表示 x1+x2^2+x1*x2
X[:,0]=x1[:,0]
X[:,1]=x2[:,0]*x2[:,0]
X[:,2]=np.multiply(x1[:,0],x2[:,0])
# 产生随机误差
bias=(np.random.rand(100,1)-0.5 )*0.4*scale
Z=X[:,0]+X[:,1]+bias[:,0]+X[:,2]
Xt,Xtest,Zt,Ztest=train_test_split(X,Z,test_size=0.4)

# 加载最小二乘法回归函数
lr=limd.LinearRegression()
print "拟合结果\n",lr.fit(Xt,Zt) # 数据进行拟合
print "系数\n",lr.coef_
print "验证结果\n",lr.score(Xtest,Ztest) # 用验证数据进行模型评分

# pyplot 三维图片加载
fig=plt.figure()
ax=aa(fig)

# plot_surface 部分
# 在数据范围内生成 meshgrid 网格坐标 100*100
Xlr=np.linspace(np.min(x1[:,0]),np.max(x1[:,0]),100)
Ylr=np.linspace(np.min(x2[:,0]),np.max(x2[:,0]),100)
[XLR,YLR]=np.meshgrid(Xlr,Ylr)
# 在网格上计算模型的输出值
i=0
ZLR=np.ones([100,100]) 
for x in Xlr:
    j=0
    for y in Ylr:
        # 对每一个坐标点作为模型的输入求解输出值 ZLR[i,j]
        ZLR[i,j]=np.dot([x,y*y,x*y],lr.coef_)+lr.intercept_
        j=j+1
    i=i+1
ax.plot_surface(XLR,YLR,ZLR,cmap='winter')  # 绘制模型输出矩阵的曲面
Zlr=np.dot(X,lr.coef_)+lr.intercept_ # 根据矩阵X计算模型输出
ax.scatter(x1,x2,Z,c='r') # 根据数据的散点图
ax.scatter(x1,x2,Zlr,c='g') # 根据模型的散点图
#ax.scatter(x1,x2,Ztest,c='r') # 根据验证数据的散点图
# 设置视觉范围
ax.view_init(elev=45,azim=-60)
# 设置 x y z 取值范围
ax.set_xlim([np.min(X[:,0]),np.max(X[:,0])])
ax.set_ylim([np.min(X[:,1]),np.max(X[:,1])])
ax.set_zlim([np.min(Z),np.max(Z)])
# 设置坐标 label
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('z')
# 展示
plt.show()
'''

'''
实例3.2.1 回归预测房价
'''

def predict_boston():
    print "回归预测房价"
    # 加载 boston 数据
    boston=lb()
    # 输出数据特征
    print boston['DESCR']
    # 验证集合样本大小占比
    cratio=0.4
    # 输出数据主键
    print boston.keys()
    # 数据分成验证集和训练集
    X,Xtest,Y,Ytest=cv.train_test_split(boston['data'],boston['target'],test_size=cratio)
    # 加载数据归一化
    S=ss()
    X=S.fit_transform(X) # 归一化拟合
    Xtest=S.fit_transform(Xtest) # 归一化拟合
    # 加载岭回归
    lr=limd.Ridge()
    # 模型训练
    model=lr.fit(X,Y)
    print "模型\n",model
    print ("训练拟合评分\n %.3f" % lr.score(X,Y))
    # 模型预测
    Ypred=model.predict(Xtest)
    print ("预测均方误差\n %.3f" % metrics.mean_squared_error(Ytest,Ypred))
    print ("系数\n %s " % lr.coef_)
    print "截距\n",lr.intercept_
    # 作图
    fig=plt.figure()
    # 画出直线 y=x 并且设置颜色和粗细
    plt.plot([min(Ytest),max(Ytest)],[min(Ytest),max(Ytest)],'r',lw=5)
    # 设置颜色
    color=abs(Ypred-Ytest)/Ytest
    # 画出散点图
    p=plt.scatter(Ypred,Ytest,c=color,marker='.')
    # 颜色刻度
    plt.colorbar()
    plt.ylabel("Predieted Price")
    plt.xlabel("Real Price")
    # 图片显示
    plt.show()
    return




'''实例3.2.2 多项式回归预测房价
'''

def poly_predict_boston():
    print "多项式回归预测房价"
    # 加载特征值预处理对象 转换成 n-degree 多项式
    pf=pp.PolynomialFeatures(2)
    # 特征值处理
    X=pf.fit_transform(X)
    Xtest=pf.fit_transform(Xtest)
    # 加载岭回归
    lr=limd.Ridge()
    # 模型训练
    model=lr.fit(X,Y)
    print "模型\n",model
    print ("训练拟合评分\n %.3f" % lr.score(X,Y))
    Ypred=model.predict(Xtest)
    print ("预测均方误差\n %.3f" % metrics.mean_squared_error(Ytest,Ypred))
    print ("系数\n %s " % lr.coef_)
    print "截距\n",lr.intercept_
    # 作图
    fig=plt.figure()
    # 画出直线 y=x 并且设置颜色和粗细
    plt.plot([min(Ytest),max(Ytest)],[min(Ytest),max(Ytest)],'r',lw=5)
    # 设置颜色
    color=abs(Ypred-Ytest)/Ytest
    # 画出散点图
    p=plt.scatter(Ypred,Ytest,c=color,marker='.')
    # 颜色刻度
    plt.colorbar()
    plt.ylabel("Predieted Price")
    plt.xlabel("Real Price")
    # 图片显示
    plt.show()
    return




'''实例3.3 LDA and QDA
'''
# 载入 python 自带鸢尾花分类问题数据集
from sklearn.datasets import load_iris as li
import sklearn.datasets as ds 
# 加载 python LDA 模块
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# 加载 python QDA 模块
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# 分类数据集 iris 加载
dataLI=li()
print "DATA 描述\n",dataLI['DESCR']

''' LDA 求解分类模型
'''
# 加载 LDA
lda=LDA()
# 随机拆分 验证集和训练集
cratio=0.4
X,Xtest,Y,Ytest=cv.train_test_split(dataLI['data'],dataLI['target'],test_size=cratio)
# LDA 拟合数据训练模型
model=lda.fit(X,Y)
print 'LDA 模型\n',model
# 预测验证集
Ypred=model.predict(Xtest)
# 预测结果评估
print "预测均方差\n",metrics.mean_squared_error(Ytest,Ypred)
print "评分\n",model.score(Xtest,Ytest)

# name 为特征名集
name=dataLI['feature_names']

# sz 特征数
sz=len(name)

# 作图1
fig=plt.figure()
# 作出两两特征为轴的平面空间内的散点分布图--关于验证集 Xtest, Ypred
for i in range(len(name)):
    for j in range(len(name)):
        # 作出子图
        ax=plt.subplot(sz,sz,i*sz+j+1)
        if i!=j:
            # 加入散点
            ax.scatter(Xtest[:,i],Xtest[:,j],s=10,c=Ypred,marker='o')
            # 设置轴的范围
            ax.set_xlim(min(Xtest[:,i]),max(Xtest[:,i]))
            ax.set_ylim(min(Xtest[:,j]),max(Xtest[:,j]))
           
        else:
            # 加入属性名称
            name[i]=name[i].rstrip('(cm)')
            ax.text(0.1,0.5,name[i])

# 作图2
fig2=plt.figure(figsize=(8,8))
# 作出两两特征为轴的平面空间内的散点分布图--关于训练集 X, Y
for i in range(len(name)):
    for j in range(len(name)):
        ax=plt.subplot(sz,sz,i*sz+j+1)
        if i!=j:
                        
            sublda=LDA()
            submodel=sublda.fit(np.c_[X[:,i],X[:,j]],Y)
            
            N=100
            xx=np.linspace(min(Xtest[:,i]),max(Xtest[:,i]),N)
            yy=np.linspace(min(Xtest[:,j]),max(Xtest[:,j]),N)
            [Xx,Yy]=np.meshgrid(xx,yy)
            
            # layout 调整
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            # 对网格点预测结果
            Z = submodel.predict(np.c_[Xx.ravel(), Yy.ravel()])
            Z = Z.reshape(Xx.shape) # 调整矩阵形状
            # 绘制等高线
            cs = plt.contourf(Xx, Yy, Z, cmap=plt.cm.RdYlBu)
            # 绘制数据散点
            
            ax.scatter(X[:,i],X[:,j],s=10,c=Y,marker='o')
            ax.set_xlim(min(X[:,i]),max(X[:,i]))
            ax.set_ylim(min(X[:,j]),max(X[:,j]))
        else:
            name[i]=name[i].rstrip('(cm)')
            ax.text(0.5,0.5,name[i],fontsize=25,
                horizontalalignment='center',
                verticalalignment='center')
plt.show()

'''QDA 求解分类模型
'''

# 加载 LDA
qda=QDA()
# QDA 拟合数据训练模型
model=qda.fit(X,Y)
print 'QDA 模型\n',model
# 预测验证集
Ypred=model.predict(Xtest)
# 预测结果评估
print "预测均方差\n",metrics.mean_squared_error(Ytest,Ypred)
print "评分\n",model.score(Xtest,Ytest)

# name 为特征名集
name=dataLI['feature_names']

# sz 特征数
sz=len(name)

# 作图3
fig3=plt.figure()
# 作出两两特征为轴的平面空间内的散点分布图--关于验证集 Xtest, Ypred
for i in range(len(name)):
    for j in range(len(name)):
        # 作出子图
        ax=plt.subplot(sz,sz,i*sz+j+1)
        if i!=j:
            # 加入散点
            ax.scatter(Xtest[:,i],Xtest[:,j],s=10,c=Ypred,marker='o')
            # 设置轴的范围
            ax.set_xlim(min(Xtest[:,i]),max(Xtest[:,i]))
            ax.set_ylim(min(Xtest[:,j]),max(Xtest[:,j]))
        else:
            # 加入属性名称
            name[i]=name[i].rstrip('(cm)')
            ax.text(0.1,0.5,name[i])

# 作图4
fig4=plt.figure(figsize=(5,5))
# 作出两两特征为轴的平面空间内的散点分布图--关于训练集 X, Y
for i in range(len(name)):
    for j in range(len(name)):
        ax=plt.subplot(sz,sz,i*sz+j+1)
        if i!=j:
                        
            sublda=QDA()
            submodel=sublda.fit(np.c_[X[:,i],X[:,j]],Y)
            
            N=100
            xx=np.linspace(min(Xtest[:,i]),max(Xtest[:,i]),N)
            yy=np.linspace(min(Xtest[:,j]),max(Xtest[:,j]),N)
            [Xx,Yy]=np.meshgrid(xx,yy)
            
            # layout 调整
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            # 对网格点预测结果
            Z = submodel.predict(np.c_[Xx.ravel(), Yy.ravel()])
            Z = Z.reshape(Xx.shape) # 调整矩阵形状
            # 绘制等高线
            cs = plt.contourf(Xx, Yy, Z, cmap=plt.cm.RdYlBu)
            # 绘制数据散点
            
            ax.scatter(X[:,i],X[:,j],s=10,c=Y,marker='o')
            ax.set_xlim(min(X[:,i]),max(X[:,i]))
            ax.set_ylim(min(X[:,j]),max(X[:,j]))
        else:
            name[i]=name[i].rstrip('(cm)')
            ax.text(0.5,0.5,name[i],fontsize=25,
                horizontalalignment='center',
                verticalalignment='center')
plt.show()



