import numpy as np
from scipy.linalg import qr
from sklearn.decomposition import PCA
from scipy.stats import ortho_group
import random
from matplotlib.pyplot import subplots, show
import matplotlib.pyplot as plt

n = 1000
m = 100
mu, sigma = 0, 3
# creating disguid data
def createData(L, m):
    mean = np.zeros(m)
    Q = ortho_group.rvs(dim=m)
    C = Q @ L @ Q.T
    X = np.random.multivariate_normal(mean, C, n)
    noise = np.random.normal(mu, sigma, [n,m])
    
    return X, X + noise, C

# drawing the graphs of error when the number of attributes increase
ax_attribute = []
pcaError_attribute = []
beError_attribute = []
for p in range(1, np.int(m/5)):
    p *= 5
    pl = np.array(np.diag(np.eye(5) * 400))
    npl = np.array(np.diag(np.eye(p)) * 5)
    l = np.append(pl,npl)
    key = l.argsort()
    l = l[key[::-1]]
    L = np.diag(l)
    X, Y, C = createData(L, 5 + p)
    
    ax_attribute.append(p)
    
    # drawing the graphs by PCA based data reconstruction
    E, V = np.linalg.eigh(C)
    key = np.argsort(E)[::-1][:5]
    E, V = E[key], V[:, key]
    pcaX = Y @ V @ V.T
    e = ((X - pcaX) ** 2).mean()
    pcaError_attribute.append(e)
    
    # drawing the graphs by PCA based data reconstruction
    beC = C - np.diag(np.diag(np.eye(5 + p)) * sigma)
    invC = np.linalg.inv(beC)
    pre = np.linalg.inv( invC + 1/sigma*np.eye(5 + p))
    aft = invC @ Y.mean(0) + Y / sigma
    beX = pre @ aft.T
    e = ((X - beX.T) ** 2).mean()
    beError_attribute.append(e)

# drawing the graphs of error when the number of principal components increase
ax_principal = []
pcaError_principal = []
beError_principal = []
for p in range(1, np.int(m/5)):
    p *= 5
    pl = np.array(np.diag(np.eye(p) * 400))
    npl = np.array(np.diag(np.eye(m - p) * 5))
    l = np.append(pl,npl)
    key = l.argsort()
    l = l[key[::-1]]
    L = np.diag(l)
    X, Y, C = createData(L, m)
    
    ax_principal.append(p)
    
    # drawing the graphs by PCA based data reconstruction
    E, V = np.linalg.eigh(C)
    key = np.argsort(E)[::-1][:p]
    E, V = E[key], V[:, key]
    pcaX = Y @ V @ V.T
    e = ((X - pcaX) ** 2).mean()
    pcaError_principal.append(e)
    
    # drawing the graphs by PCA based data reconstruction
    beC = C - np.diag(np.diag(np.eye(m)) * sigma)
    invC = np.linalg.inv(beC)
    pre = np.linalg.inv( invC + 1/sigma*np.eye(m))
    aft = invC @ Y.mean(0) + Y / sigma
    beX = pre @ aft.T
    e = ((X - beX.T) ** 2).mean()
    beError_principal.append(e)

# drawing the graphs of error when the values of the non-principal increase
ax_nonprin = []
pcaError_nonprin = []
beError_nonprin = []
for p in range(6, 20):
    pl = np.array(np.diag(np.eye(20)) * 400)
    npl = np.array(np.diag(np.eye(m - 20)) * p)
    l = np.append(pl,npl)
    key = l.argsort()
    l = l[key[::-1]]
    L = np.diag(l)
    X, Y, C = createData(L, m)
    
    ax_nonprin.append(p)
    
    # drawing the graphs by PCA based data reconstruction
    E, V = np.linalg.eigh(C)
    key = np.argsort(E)[::-1][:20]
    E, V = E[key], V[:, key]
    pcaX = Y @ V @ V.T
    e = ((X - pcaX) ** 2).mean()
    pcaError_nonprin.append(e)
    
    # drawing the graphs by PCA based data reconstruction
    beC = C - np.diag(np.diag(np.eye(m)) * sigma)
    invC = np.linalg.inv(beC)
    pre = np.linalg.inv( invC + 1/sigma*np.eye(m))
    aft = invC @ Y.mean(0) + Y / sigma
    beX = pre @ aft.T
    e = ((X - beX.T) ** 2).mean()
    beError_nonprin.append(e)

plt.plot(ax_attribute, pcaError_attribute, 'p-', ax_attribute, beError_attribute, 'ro-')
plt.gca().legend(('PCA-DR Scheme','BE-DR Scheme'))
plt.title('Increase the Number of Attributes')
plt.xlabel('Number of Attributes')
plt.ylabel('Root Mean Square Error')
plt.show()

plt.plot(ax_principal, pcaError_principal, 'p-', ax_principal, beError_principal, 'ro-')
plt.gca().legend(('PCA-DR Scheme','BE-DR Scheme'))
plt.title('Increase the Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Root Mean Square Error')
plt.show()

plt.plot(ax_nonprin, pcaError_nonprin, 'p-', ax_nonprin, beError_nonprin, 'ro-')
plt.gca().legend(('PCA-DR Scheme','BE-DR Scheme'))
plt.title('Increase the Eigenvalues of the non-Principal Components')
plt.xlabel('Eigenvalues of the non-Principal Components')
plt.ylabel('Root Mean Square Error')
plt.show()
