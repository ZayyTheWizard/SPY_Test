import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train():
    # Data Preprocessing 
    df = pd.read_csv("./Data/DataProccessed.csv")

    x = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=0.2, random_state=0)

    sc_x = StandardScaler()

    X_Train = sc_x.fit_transform(X_Train)
    X_Test = sc_x.transform(X_Test)

    sc_y = StandardScaler()
    Y_Train = sc_y.fit_transform(Y_Train.reshape(-1, 1)).ravel()
    Y_Test = sc_y.transform(Y_Train.reshape(-1, 1)).ravel()

    ann = tf.keras.models.Sequential()

    ann.add(tf.keras.layers.Dense(units=12, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=12, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=12, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1))

    ann.compile(optimizer='adam', loss='mean_squared_error')

    ann.fit(X_Train, Y_Train, batch_size=32, epochs=20000)

    ann.save('TrainedSPY')

    pred = ann.predict(sc_x.transform([[468.30, 470.96, 467.05, 84230214]]))
    print(sc_y.inverse_transform(pred))



if __name__ == '__main__':
    train()