# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:37:53 2018

@author: lenovo
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn import datasets as DS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split


dg=DS.load_digits()
image=dg['images']
target=dg['target']
data=dg['data']
[X,Xtest,Y,Ytest]=train_test_split(data,target,test_size=0.3)
plt.imshow(image[0])

'''LDA
'''
lda=LDA()
model=lda.fit(X,Y)
print model.score(X,Y),model.score(Xtest,Ytest)
y=model.predict(Xtest)
print target[y!=target]

'''DecisionTree
'''
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import export_graphviz as GV
dt=DTC()
model=dt.fit(X,Y)
print model.score(X,Y),model.score(Xtest,Ytest)
y=model.predict(Xtest)
print target[y!=target]
print model.tree_


from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D as aa

[X,Y]=make_classification(n_samples=100000,n_classes=4,n_informative=3,n_features=10,
                           n_redundant=3,n_repeated=3)

[X,Xtest,Y,Ytest]=train_test_split(X,Y,test_size=0.3)

pca=PCA(n_components=3)
Xtr=pca.fit_transform(X,Y)
print pca.explained_variance_ratio_
#Xtesttr=pca.fit_transform(Xtest,Ytest)
dtc=DTC()
model=dtc.fit(Xtr,Y)

fig1=plt.figure()

plt.scatter(Xtr[:,0],Xtr[:,1],c=Y,s=20)

plt.show()

fig=plt.figure()
ax=aa(fig)

ax.scatter(Xtr[:,0],Xtr[:,1],Xtr[:,2],c=Y,s=20)

plt.show()