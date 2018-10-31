# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 12:10:43 2018

@author: lenovo
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import datasets
from sklearn.datasets import load_iris as iris
from sklearn.model_selection import KFold
from sklearn import cross_validation as CV
from sklearn import tree
from mpl_toolkits.mplot3d import Axes3D as aa


def guassian_classification(method):
    N=100
    X=sp.randn(N,2)
    X[0:N/2-1,:]=(X[0:N/2-1,:]-1)/1
    X[N/2:N,:]=(X[N/2:N,:]+1)/1
    Y=np.zeros(N)
    Y[0:N/2-1]=1
    Y[N/2:N]=-1
    treeclr=tree.DecisionTreeClassifier(criterion='entropy')
    model=treeclr.fit(X,Y)
    print model
    # plot the decision boundary
    num=100
    [minx1,maxx1]=[min(X[:,0]),max(X[:,0])]
    [minx2,maxx2]=[min(X[:,1]),max(X[:,1])]
    fig=plt.figure()
    [XX,YY]=np.meshgrid(np.linspace(minx1,maxx1,num),np.linspace(minx2,maxx2,num))
    Z=model.predict(np.c_[XX.ravel(), YY.ravel()])
    plt.tight_layout(h_pad=0.05, w_pad=0.05, pad=2.5)
    Z=Z.reshape(XX.shape)
    cs = plt.contourf(XX, YY, Z, cmap=plt.cm.RdYlBu)
    fig.hold()
    plt.scatter(X[:,0],X[:,1],5,Y)
    fig.show()
    # plot the matrix Z
    fig2=plt.figure()
    [xsize,ysize]=Z.shape
    [Xx,Yy]=np.meshgrid(np.linspace(0,xsize-1,xsize),np.linspace(0,ysize-1,ysize))
    plt.scatter(Xx,Yy,c=Z)
    fig2.show()
    print Z
    return
#guassian_classification('entropy')

def guassian_regression():
    N=100
    X=np.linspace(1,N,N)
    Y=np.log10(np.power(X,2))+sp.randn(1,N)/4

    treereg=tree.DecisionTreeRegressor()
    model=treereg.fit(X,Y)
    print model
    
    num=100
    x=np.linspace(1,num,num)
    x=x.reshape(num,1)
    y=model.predict(x)
    print x,y
    fig=plt.figure()
    X=X.reshape(1,N)
    plt.scatter(X,Y)
    fig.hold()
    x=x.reshape(1,num)
    plt.plot(x,y)
    fig.show()
    return


from sklearn.model_selection import train_test_split
#def wine_classification():
# 打开数据描述文件
f=open('C:\Users\lenovo\Desktop\wine_names.csv')
names=f.read(10000)
# 打开数据内容文件
data=pd.read_csv('C:\Users\lenovo\Desktop\wine_data.csv')
print names,data
data=data.values
Num=data.shape[0] #获取数据的数量
FeatureNum=data.shape[1]-1 #获取数据的特征数
# 将数据分成 data & target 两部分
Y=data[:,0]
X=data[:,1:FeatureNum+1]
[X,Xtest,Y,Ytest]=train_test_split(X,Y,test_size=0.3,shuffle=True)
F=plt.figure(figsize=(30,30))

def compute_example():
    X=[[1,0,125],[0,1,100],[0,0,70],[1,1,120],[0,2,95],
       [0,1,60],[1,2,220],[0,0,85],[0,1,75],[0,0,90]]
    X=np.array(X)
    Y=[0,0,0,0,1,0,0,1,0,1]
    Y=np.array(Y)
    Y=Y.reshape(10,1)
    Xtest=X[8:10]
    X=X[0:8]
    Ytest=Y[8:10]
    Y=Y[0:8]
    p1=sum(Y==0)
    p2=sum(Y==1)
    p1=p1/(p1+p2)
    p2=p2/(p1+p2)
    entD=-(p1*np.log(p1)/np.log(2)+p2*np.log(p2)/np.log(2))



# plot distribution
def plot_distribution(X,Y,FeatureNum):
    for i in range(FeatureNum):
        for j in range(FeatureNum):
            fig=plt.subplot(FeatureNum,FeatureNum,i*FeatureNum+j+1)
            if i!=j:
                fig.scatter(X[:,i],X[:,j],s=15,c=Y)
                fig.set_xlim(min(X[:,i]),max(X[:,i]))
                fig.set_ylim(min(X[:,j]),max(X[:,j]))
            else:
                label='Feature'+str(i+1)
                fig.text(0.5, 0.5,label, fontsize=25,
                horizontalalignment='center',
                verticalalignment='center')
    plt.show()
    return


# plot contour surface
def plot_contour_surface(X,Y,FeatureNum):
    # 定义figure 大小30*30
    Fig=plt.figure(figsize=(30,30))
    # 对特征两两组合
    for i in range(FeatureNum):
        for j in range(FeatureNum):
            # 作出子图
            fig=plt.subplot(FeatureNum,FeatureNum,i*FeatureNum+j+1)
            if i!=j:
                # 对特征模型拟合
                dtc=DTC(criterion='gini')
                submodel=dtc.fit(np.c_[X[:,i],X[:,j]],Y)
                # 形成平面内的网格点
                num=100 # 数量
                [minx1,maxx1]=[min(X[:,i]),max(X[:,i])]
                [minx2,maxx2]=[min(X[:,j]),max(X[:,j])]
                [xx,yy]=np.meshgrid(np.linspace(minx1,maxx1,num),
                                    np.linspace(minx2,maxx2,num))
                # layout 调整
                plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
                # 对网格点预测结果
                Z = submodel.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape) # 调整矩阵形状
                # 绘制等高线
                cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
                # 绘制数据散点
                fig.scatter(X[:,i],X[:,j],s=15,c=Y,cmap=plt.cm.RdYlBu, edgecolor='black')
                fig.set_xlim(min(X[:,i]),max(X[:,i]))
                fig.set_ylim(min(X[:,j]),max(X[:,j]))
            else:
                label='Feature'+str(i+1)
                # 添加label
                fig.text(0.5, 0.5,label, fontsize=25,
                horizontalalignment='center',
                verticalalignment='center') # 居中
    plt.show()
    return
'''回归时候用
'''
# measured-predicted curve
def measured_predicted_curve(model,X,Xtest,Y,Ytest):
    Ypred=model.predict(Xtest)
    fig=plt.figure(figsize=(10,10))
    N=100
    plt.plot(np.linspace(min(Ypred),max(Ypred),N),np.linspace(min(Ypred),max(Ypred),N),'r')
    plt.scatter(Ytest,Ypred)
    plt.show()
    return

#plot_distribution(X,Y,FeatureNum)
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import export_graphviz as GV
# fit data
# 利用method criterion来完成决策树的分类
def fit_wine_data(method):
    # 定义一个DecisionTree分类器 以method方式
    dtc=DTC(criterion=method)
    # 对数据进行拟合
    model=dtc.fit(X,Y)
    
    # 画出决策树
    gv=GV(model,out_file='C:\Users\lenovo\Desktop\winetree.dot',feature_names=['feature1',
                         'feature2','feature3','feature4','feature5','feature6','feature7',
                         'feature8','feature9','feature10','feature11',
                         'feature12','feature13'],
                         class_names=['class1','class2','class3'],rounded=True)
    print gv
    
    # 计算model拟合的训练评分
    train_score=model.score(X,Y)
    # 计算model验证评分
    test_score=model.score(Xtest,Ytest)
    print('train score %.8f \ntest score %.8f\n' %(train_score ,test_score))  # 输出结果
    return [train_score,test_score]

[trains,tests]=fit_wine_data('entropy')


# para1=max_depth 对数据进行拟合
def fit_wine_data_depth(method,para1):
    # 数据拟合
    dtc=DTC(criterion=method,max_depth=para1)
    model=dtc.fit(X,Y)
    # 模型训练评分
    train_score=model.score(X,Y)
    # 模型拟合评分
    test_score=model.score(Xtest,Ytest)
    print('train score %.8f \n test score %.8f\n' %(train_score ,test_score))  # 格式化输出
    return [train_score,test_score]

# 寻找最优max_depth并绘制曲线
def plot_wine_data_depth():
    N=100 # 点数量
    depths=np.linspace(2,200,N)
    fig=plt.figure(figsize=(10,10))
    trainsc=np.zeros(N)
    testsc=np.zeros(N)
    
    # 对depths中所有参数进行拟合
    for i in range(N):
        # 返回评分结果
        x=fit_wine_data_depth('entropy',depths[i])
        trainsc[i]=x[0]
        testsc[i]=x[1]
        
    # 绘制评分曲线
    plt.plot(depths,trainsc,'.r--',label='train score')
    plt.plot(depths,testsc,'g',label='test score')
    plt.scatter(depths,testsc,c='y')
    # 图像基本设置
    plt.grid()
    plt.title('train score & test score related with max_depth')
    plt.xlabel('max_depth')
    plt.ylabel('score %')
    plt.legend()
    plt.show()
    
    # 计算最优参数
    r=np.max(testsc)
    idx=np.where(testsc==r)
    print r,idx
    return [r,idx]
# para1=min_impurity_decrease 数据拟合
def fit_wine_data_impurity(method,para1):
    # 数据拟合
    dtc=DTC(criterion=method,min_impurity_decrease=para1)
    model=dtc.fit(X,Y)
    # 模型训练评分
    train_score=model.score(X,Y)
    # 模型拟合评分
    test_score=model.score(Xtest,Ytest)
    return [train_score,test_score]

def plot_wine_data_impurity():
    N=100
    impurity=np.linspace(0,0.2,N)
    fig2=plt.figure(figsize=(10,10))
    trainsc2=np.zeros(N)
    testsc2=np.zeros(N)
    for i in range(N):
        x=fit_wine_data_impurity('gini',impurity[i])
        trainsc2[i]=x[0]
        testsc2[i]=x[1]
    plt.plot(impurity,trainsc2,'.r--',label='train score')
    plt.plot(impurity,testsc2,'g',label='test score')
    plt.scatter(impurity,testsc2,c='y')
    plt.title('train score & test score related with max_depth')
    plt.grid()
    plt.xlabel('min impurity')
    plt.ylabel('score %')
    plt.legend()
    plt.show()
    r=np.max(testsc2)
    idx=np.where(testsc2==r)
    print r,idx
    return [r,idx]

#plot_wine_data_depth()  
#plot_wine_data_impurity()

# hyperparameter optimization
from sklearn.model_selection import GridSearchCV as GSCV

def GridSearchPara():
    N=100
    impurity=np.linspace(0,0.2,N)
    hyperpara={'criterion':['gini','entropy'],
               'min_impurity_decrease':impurity,
               'max_depth':np.linspace(1,200,N)}
    model=GSCV(DTC(),hyperpara,cv=5)
    model.fit(X,Y)
    print model.best_params_,model.best_score_
    return
GridSearchPara()
# 利用 GridSearchCV 对单个参数进行优化
def GridSearchSinglePara():
    N=50 # 点数量
    max_depth=np.linspace(1,200,N)
    hyperpara={'max_depth':max_depth} # 参数字典
    # GridSearchCV 对象
    model=GSCV(DTC(),hyperpara,cv=5)
    model.fit(X,Y)
    print model.best_params_,model.best_score_
    # 作图部分
    R=model.cv_results_
    # 平均训练评分
    mtrains=R['mean_train_score']
    # 标准训练评分
    strains=R['std_train_score']
    # 平均验证评分
    mtests=R['mean_test_score']
    # 标准验证评分
    stests=R['std_test_score']
    # 作图
    fig=plt.figure(figsize=(10,10))
    # 填充
    plt.fill_between(max_depth,mtrains-strains,
                     mtrains+strains,color='lightgray',alpha=0.3)
    plt.fill_between(max_depth,mtests-stests,
                     mtests+stests,color='lightgray',alpha=0.3)
    # 曲线
    plt.plot(max_depth,mtrains,color='r',label='train mean scores')
    plt.plot(max_depth,mtests,color='g',label='test mean scores')
    # 图的基本设置
    plt.grid()
    plt.legend()
    plt.title('max_depth gridsearch')
    plt.xlabel('max_depth')
    plt.ylabel('score %')
    plt.show()
#GridSearchSinglePara()
    