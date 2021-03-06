# Nyström Accelerated Robust Spectral Clustering

SCAR is a python library for implementing the Nyström Accelerated Robust Spectral clustering method.

###  Set Up
SCAR requires the NumPy, SciPy, pandas, and SciKit-Learn package to operate.

### Run
To call SCAR for a dataset (e.g. moon) use:
```python
from sklearn.datasets import make_moons
from SpectralClusteringAcceleratedRobust import SCAR

X, y = make_moons(n_samples=1000, noise=0.09)

k = 2  # number of clusters
theta = 20  # max number of corrupted edges to remove
alpha = 0.2  # portion of landmark points
num_neighbours = 45  # number of neighbours

""" run RSC accelerated with Nyström (no further modifications) """
scar = SCAR(k=k, nn=num_neighbours, alpha=alpha, theta=theta, laplacian=2).fit_predict(X)
""" calculate robust clustering with Nystrom and other improvements"""
scar = SCAR(k=k, nn=num_neighbours, alpha=alpha, theta=theta, laplacian=0, normalize = True, weighted = True).fit_predict(X)
```

### Licence
[MIT](https://choosealicense.com/licenses/mit/)
