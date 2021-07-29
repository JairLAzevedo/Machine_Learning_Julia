import numpy as np
import matplotlib.pyplot as plt
import csv

from numpy import genfromtxt

my_data = genfromtxt('df_data.csv', delimiter=',')
X = np.ones((np.size(my_data,axis=0)-1,3))
y = 3*np.ones((np.size(my_data,axis=0)-1,1))
for i in range(0,(np.size(my_data,axis=0)-1)):
        X[i][:]=my_data[i+1][0:3]
    
with open('df_data.csv') as csvfile: 
        my_data_2 = csv.reader(csvfile)  
        i=0
        next(my_data_2)
        for line in my_data_2:
            if line[3]== 'med':
                y[i]=0
            elif line[3]=='low':
                y[i]=-1
            else:
                y[i]=1
            i=i+1



from lda import LDA
lda1 = LDA(3)
lda1.fit(X,y)
X_proj = lda1.transform(X)

X1 = X_proj[:,0]
X2 = X_proj[:,1]
X3 = X_proj[:,2]

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
c = np.arange(len(X)) / len(X)  # create some colours
p = ax.scatter(X1, X2, X3, c=y,alpha=0.8, cmap=plt.cm.get_cmap('viridis',3))
fig.colorbar(p)
plt.show()

#------------ 2D plots --------------------------------------------

plt.scatter(X1,X2,c=y,edgecolor='none',alpha=0.8,cmap=plt.cm.get_cmap('viridis',3))
plt.ylabel('Linear Discriminant 2')
plt.xlabel('Linear Discriminant 1')
plt.colorbar()
plt.show()

plt.scatter(X1,X3,c=y,edgecolor='none',alpha=0.8,cmap=plt.cm.get_cmap('viridis',3))
plt.ylabel('Linear Discriminant 3')
plt.xlabel('Linear Discriminant 1')
plt.colorbar()
plt.show()

plt.scatter(X2,X3,c=y,edgecolor='none',alpha=0.8,cmap=plt.cm.get_cmap('viridis',3))
plt.ylabel('Linear Discriminant 3')
plt.xlabel('Linear Discriminant 2')
plt.colorbar()
plt.show()