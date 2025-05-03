import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
X = df.drop('Y', axis=1)
y = df['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")

# Mean Squared Error: 2900.1936
w = model.coef_
intercept = model.intercept_

print("Trọng số (w):", w)
print("Hệ số chệch (intercept):", intercept)

# Trọng số (w): [  0.13768782 -23.06446772   5.84636265   1.19709252  -1.28168474
#    0.81115203   0.60165319  10.15953917  67.1089624    0.20159907]
# Hệ số chệch (intercept): -341.378236333505