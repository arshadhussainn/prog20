import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X,y = load_iris(return_X_y=True)

model = LogisticRegression(max_iter=200)
model.fit(X,y)

with open("iris_model.pkl","wb") as f:
    pickle.dump(model,f)

print("model trained and saved successfully")