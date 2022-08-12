# SCAR – Spectral Clustering Accelerated & Robust

SCAR is a python library for implementing a Nyström-accelerated and robust spectral clustering method. Its implementation and experiments are described in [this paper](https://doi.org/10.14778/3551793.3551850).

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

### Experiments
SCAR was tested on 2 synthetic and 7 real-world datasets. 
Both synthetic datasets, *moons* and *circles*, were constructed using data generator functions from the [scikit-learn library](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets). 
Real world benchmark datasets *iris*, *dermatology*, *banknote*, *pendigits*, and *Letter Recognition* (letters for short) were obtained from the [UCI- MLR1](https://archive.ics.uci.edu/ml/index.php). *MNIST* and *USPS* were obtained from the repository of the [CS NYU 2](https://cs.nyu.edu/~roweis/data.html). 

The synthetic datasets were created with n=1000 and noise=0.15.
For dermatology we omit the feature about the age of patients as the dataset is incomplete w.r.t this feature. 
For all other datasets, we used the raw data as provided by the respective repositories as input features without any preprocessing step.

For transparency, further information on how SCAR scored on the tested datasets w.r.t. its hyperparameter settings can be found in [HyperparameterSettings.txt](HyperparameterSettings.txt) or in the paper. 

### Licence
[MIT](https://choosealicense.com/licenses/mit/)
