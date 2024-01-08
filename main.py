import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

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

    ann.fit(X_Train, Y_Train, batch_size=32, epochs=50000)

    ann.save('TrainedSPY')
    joblib.dump(sc_x, 'sc_x_scaler.save')
    joblib.dump(sc_y, 'sc_y_scaler.save')

    pred = ann.predict(sc_x.transform([[468.30, 470.96, 467.05, 84230214]]))
    print(sc_y.inverse_transform(pred))

def predict(open: float, high: float, low: float, vol: int):
    sc_x = joblib.load('sc_x_scaler.save')
    sc_y = joblib.load('sc_y_scaler.save')
    loaded_ann = tf.keras.models.load_model('TrainedSPY')

    input_data = np.array([[open, high, low, vol]])
    pred = loaded_ann.predict(sc_x.transform(input_data))

    return sc_y.inverse_transform(pred)

if __name__ == '__main__':
    while(True):
        open_price: float = input("open price?: \n")
        high_price: float = input("high price?: \n")
        low_price: float = input("low price?: \n")
        vol: int = input("open vol?: \n")

        prediction = predict(open_price, low_price, high_price, vol)
        print(prediction)

        choice = input("Would you like to continue y/n: ")

        if choice == 'y':
            continue
        elif choice == 'n':
            break
        elif choice != 'y' or choice != 'n':
            break
