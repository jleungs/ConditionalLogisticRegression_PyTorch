# ConditionalLogisticRegression-PyTorch
Conditional/Fixed-Effects Logistic Regression in PyTorch with Mini-Batch Gradient Descent optimization.

## Example usage
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
model = ConditionalLogisticRegression(X_normalized, y, groups)
model.fit()
```
