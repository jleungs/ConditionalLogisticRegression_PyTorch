# ConditionalLogisticRegression_PyTorch
Conditional/Fixed-Effects Logistic Regression in PyTorch with Mini-Batch Gradient Descent optimization. Early stopping is supported, so if validation loss is increasing with some delta and patience, it will automatically terminate earlier.

Also, multithreaded k-fold cross-validation coupled with grid search for hyperparameter optimisation is implemented.

See below for example usage.

## Example usage
### Fitting model with synthetic data
```
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
# import this repo
from src.clogit import ConditionalLogisticRegression

# generate synthetic classification data using scikit-learn
n_samples = 100
n_features = 5
n_groups = 10

X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2)
# normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
# generate synthetic groups
groups = np.random.randint(0, n_groups, size=n_samples)
# fit the ConditionalLogisticRegression model
model = ConditionalLogisticRegression(epochs=300, l2_constant=1e-3, lr=1e-2, groups_batch_size=1, earlystop_patience=3, earlystop_delta=17)
model.fit(X_normalized, y, groups)
```

### Using grid search k-fold cross-validation
```
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
# import this repo
from src.clogit import ConditionalLogisticRegression
from src.lib import GridSearchKFoldCV

# generate synthetic classification data using scikit-learn
n_samples = 100
n_features = 5
n_groups = 10

X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2)
# normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
# generate synthetic groups
groups = np.random.randint(0, n_groups, size=n_samples)
# create parameter grid
param_grid = {"lr": [1e-1, 1e-2, 1e-3], "l2_constant": [1e-2, 1e-3, 1e-4]}
# start grid search
gs = GridSearchKFoldCV(ConditionalLogisticRegression, param_grid, n_threads=8)
gs.fit(X_normalized, y, groups)
# print best model
print("Best parameters:", gs.best_score())
```
