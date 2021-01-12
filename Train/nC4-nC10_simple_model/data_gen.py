'''

Generate all possible mixture compositions for n components

'''

import numpy as np

# Number of components
nc = 3

# Define number of points per component
n = 101

allcomps = []


for i in np.linspace(0,1,n):
    for j in np.linspace(0,1,n):
        for k in np.linspace(0,1,n):
            if (i+j+k) == 1:
                allcomps.append([i,j,k])

allcomps = np.array(allcomps)
print(allcomps)
print(allcomps.shape)
