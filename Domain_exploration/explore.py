import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

csv_path = r'E:\Datasets\data_const_T_20200716-230921.csv'
d = pd.read_csv(csv_path, delimiter=',', names=['a_mix', 'b_mix',
                                            'b_i', 'sum', 'lnphi'])
#d.describe()

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(d.drop(['lnphi'], axis=1))
X = d.drop(['lnphi'], axis=1)

# Reduce size
X_reduced_partial, X_reduced_test, y_train_partial, y_train_test = train_test_split(
            X_reduced, d[['lnphi']], test_size=0.9999, random_state=42)

X_partial, X_test, y_train_partial, y_train_test = train_test_split(
    X, d[['lnphi']], test_size=0.9999, random_state=42
)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_reduced_partial[:,0], X_reduced_partial[:,1], y_train_partial, marker='o')
ax.set_xlabel('X_1')
ax.set_ylabel('X_2')
ax.set_zlabel('lnphi')
plt.show()
'''
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(X_partial.iloc[:,0], X_partial.iloc[:,1], y_train_partial, marker='o')
ax2.set_xlabel('am')
ax2.set_ylabel('bm')
plt.show()

###################
# Check for b_mix < 0.05. Case study min was 0.01.
X2 = d[d['b_mix'] < 0.05].drop(['lnphi'], axis=1)
y2 = d[d['b_mix'] < 0.05].loc[:,'lnphi']

X_partial, X_test, y_train_partial, y_train_test = train_test_split(
    X2, y2, test_size=0.9999, random_state=42
)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(X_partial.iloc[:,0], X_partial.iloc[:,1], y_train_partial, marker='o')
ax3.set_xlabel('a_mix')
ax3.set_ylabel('b_mix')
plt.show()

####
# Limit lnphi values to [-50,50] then plot vs a_mix, b_mix
desc = d.drop(d.loc[(d.loc[:,'lnphi'] < -50) | (d.loc[:,'lnphi'] > 50)].index)
X2 = desc.drop(['lnphi'], axis=1)
y2 = desc.loc[:,'lnphi']
X_partial, X_test, y_train_partial, y_train_test = train_test_split(
    X2, y2, test_size=0.9999, random_state=42)
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.scatter(X_partial.iloc[:,0], X_partial.iloc[:,1], y_train_partial, marker='o', alpha=0.1)
ax4.set_xlabel('a_mix')
ax4.set_ylabel('b_mix')
plt.show()


# Limit lnphi values to [-50,50] then do PCA and plot vs X1 and X2.
desc = d.drop(d.loc[(d.loc[:,'lnphi'] < -50) | (d.loc[:,'lnphi'] > 50)].index)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(desc.drop(['lnphi'], axis=1))
print(pca.explained_variance_ratio_)
X_reduced_partial, X_reduced_test, y_train_partial, y_train_test = train_test_split(
            X_reduced, desc.loc[:,'lnphi'], test_size=0.9999, random_state=42)
print(X_reduced_partial.shape)
print(y_train_partial.shape)
fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')
ax5.scatter(X_reduced_partial[:,0], X_reduced_partial[:,1], y_train_partial, marker='o', alpha=0.1)
ax5.set_xlabel('X_1')
ax5.set_ylabel('X_2')
plt.show()
'''