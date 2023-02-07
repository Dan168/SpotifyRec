import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import neighbors

df = pd.read_csv('spotify_dataset.csv')

df.dropna()

X = df.iloc[:, 13:20].values
y = df['Name'].values

X = pd.DataFrame(X)
X = X.apply(pd.to_numeric, errors='coerce')

si = SimpleImputer(strategy='mean')
X = si.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Scaling seems to throw it off
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print(knn.predict([[0.83, 0.839,-6.985,0.0699,0.00201,0.0353,128.012]]))
print(knn.score(X_test, y_test))
