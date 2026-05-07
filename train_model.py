
# train_model.py
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# Dummy data
X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))