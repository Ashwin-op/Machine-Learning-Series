### Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the Dataset
dataset = pd.read_csv('Data.csv')
print("The imported dataset is:")
print(dataset)  # It is a dataframe

print("\n\n")


# Separating independent and dependent features
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("Independent features values are:")
print(X)

print("\n\n")

print("Dependent features values are:")
print(y)

print("\n\n")


# Taking care of missing data
from sklearn.impute import SimpleImputer

si = SimpleImputer(missing_values=np.nan, strategy='mean')
si.fit(X[:, 1:3])
X[:, 1:3] = si.transform(X[:, 1:3])

print("Missing values of independent features replaced with mean:")
print(X)

print("\n\n")


# Encoding categorical values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])
print("Label encoding of the country features:")
print(X)

print("\n\n")

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print("One hot encoding of the country features:")
print(X)

print("\n\n")

y = LabelEncoder().fit_transform(y)
print("Label encoding of the purchased features:")
print(y)

print("\n\n")


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)
print("After feature scaling the independent features:")
print(X)

print("\n\n")


# Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("The split sets:")
print(X_train, X_test, y_train, y_test, sep="\n\n")
