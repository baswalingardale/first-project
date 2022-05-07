import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Fetching data
df = pd.read_csv('Iris.csv')

# EDA
df.drop(['Id'], axis=1, inplace=True)

df['Species'].replace({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}, inplace=True)

# Features
x = df.iloc[:,1:]

# Targest column
y = df['SepalLengthCm']

# Spliting data to train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Training Model
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

def iris_pred(sapal_wid, petal_len, petal_wid, species):

    sapal_wid = float(sapal_wid)
    petal_len = float(petal_len)
    petal_wid = float(petal_wid)

    input_data = np.array([[sapal_wid, petal_len, petal_wid, species]])

    sapal_len = linear_model.predict(input_data)

    return sapal_len



if __name__ == "__main__":
    print(iris_pred(3.5,1.4,0.2,0))