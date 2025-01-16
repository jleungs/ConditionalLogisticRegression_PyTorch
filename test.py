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

