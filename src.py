
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
train_df = pd.read_csv("train.dat",header=None, delimiter=r"\s+", dtype=float)
train = train_df.values

train_labels_df = pd.read_csv("train.labels", header=None, delimiter=r"\s+", dtype=int)
train_labels = np.concatenate(np.array(train_labels_df.values), axis=0)



test_df = pd.read_csv("test.dat", header=None, delimiter=r"\s+", dtype=float)
test = test_df.values


# In[16]:


X_std = StandardScaler().fit_transform(train)
import numpy as np
mean_vec = np.mean(X_std, axis = 0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)


# In[18]:


cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
means = np.mean(X_std, axis=0)
X_sm = X_std - means
print('Eigenvectors \n%s' %eig_vecs)
print('Eigenvalues \n%s' %eig_vals)


# In[19]:


X_cov = X_sm.T.dot(X_sm) / (X_sm.shape[0] - 1)

# Side-note: Numpy has a function for computing the covariance matrix
X_cov2 = np.cov(X_std.T)
print("X_cov == X_cov2: ", np.allclose(X_cov, X_cov2))

# perform the eigendecomposition of the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(X_cov)
print('Eigen Vec',eig_vecs)


# In[20]:


def percvar(v):
    r"""Transform eigen/singular values into percents.
    Return: vector of percents, prefix vector of percents
    """
    # sort values
    s = np.sort(np.abs(v))
    # reverse sorting order
    s = s[::-1]
    # normalize
    s = s/np.sum(s)
    return s, np.cumsum(s)
print("eigenvalues:    ", eig_vals)
pct, pv = percvar(eig_vals)
print("percent values: ", pct)
print("prefix vector:  ", pv)


# In[21]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.plot(range(1,len(pct)+1),pct,label = "fetaure")
plt.plot(range(1,len(pv)+1),pv,label = "Overall")
plt.xlabel("K - Number of Values")
plt.ylabel("Variance - Percent Values")
plt.show()


# In[22]:


def perck(s, p):
    for i in range(len(s)):
        if(s[i]>=p):
            return i+1
    return len(s)

for p in [40, 60, 80, 90, 95]:
    print("Number of dimensions to account for %d%% of the variance: %d" % (p, perck(pv, p*0.01)))


# In[23]:


from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=10, algorithm='auto')

from sklearn import random_projection
rp = random_projection.SparseRandomProjection(n_components=32)
rp.fit(train,train_labels)

X_train_rp = rp.transform(train)
X_test_rp = rp.transform(test)

print(X_train_rp.shape)
print(X_test_rp.shape)


# In[26]:


clf.fit(X_train_rp, train_labels)
y = clf.predict(X_test_rp)

result = pd.DataFrame(y)

result.to_csv('result_finalsubmission_predicte_data.dat',index=False,header=None)

