import copy
import numpy                 as np
import pandas                as pd
import seaborn               as sns
import tensorflow            as tf
import matplotlib.pyplot     as plt
from   sklearn.linear_model  import LinearRegression
from   sklearn.preprocessing import StandardScaler


dataset_cols = ["bike_count", "hour", "temperature", "humidity", "wind", "visibility", "dew_pt_temp", "radiation", "rain", "snow", "functional"]
df = pd.read_csv("SeoulBikeData.csv").drop(["Date", "Seasons", "Holiday", "Functioning Day"], axis=1)

df.columns = dataset_cols
df["functional"] = (df["functional"] == "Yes").astype(int)
df = df[df["hour"] == 12]
df = df.drop(["hour"], axis=1)


df.head()


for label in df.columns[1:]:
    plt.scatter(df[label], df["bike_count"])
    plt.title(label)
    plt.ylabel("bike_count noon")
    plt.xlabel(label)
    plt.show()
    

df = df.drop(["wind", "visibility", "functional"], axis=1)

df.head()


# train-test valid

train, val, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])


def get_xy(dataframe, y_label, x_labels= None):
    dataframe = copy.deepcopy(dataframe)

    if not x_labels:
        X = dataframe[[c for c in dataframe.columns if c != y_label]].values

    else:
        if len(x_labels) == 1:
            X = dataframe[[x_labels]].values

        else:
            X = dataframe[x_labels].values

    y = dataframe[y_label].values.reshape(-1, 1)
    data = np.hstack((X, y))

    return data, X, y
        

_, X_train_temp, y_train_temp = get_xy(train, "bike_count", x_labels= ["temp"])
_, X_train_val,  y_train_val  = get_xy(val,   "bike_count", x_labels= ["temp"])
_, X_train_test, y_train_test = get_xy(test,  "bike_count", x_labels= ["temp"])


temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)

temp_reg.score(X_train_test, y_train_test)

# time: 2:50:00

plt.scatter(X_train_temp, y_train_temp, label= "Data", color='blue')
x = tf.linspace(-20, 40, 100)
plt.plot(x, temp_reg.predict(np.array(x).reshape(-1, 1)), label= "Fit", color= 'red', linewidth= 3)
plt.legend()
plt.title("Bikes vs Temp (NN)")
plt.ylabel("Bike Count")
plt.xlabel("Temperature (C)")
plt.show()


df.head()


# train-test valid

train, val, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])


def get_xy(dataframe, y_label, x_labels= None):
    dataframe = copy.deepcopy(dataframe)

    if x_labels is None:
        X = dataframe[[c for c in dataframe.columns if c != y_label]].values

    else:
        if len(x_labels) == 1:
            X = dataframe[[x_labels]].values

        else:
            X = dataframe[x_labels].values

    y = dataframe[y_label].values.reshape(-1, 1)
    data = np.hstack((X, y))

    return data, X, y
        

_, X_train_temp, y_train_temp = get_xy(train, "bike_count", x_labels= ["temp"])
_, X_train_val,  y_train_val  = get_xy(val,   "bike_count", x_labels= ["temp"])
_, X_train_test, y_train_test = get_xy(test,  "bike_count", x_labels= ["temp"])


temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)

temp_reg.score(X_train_test, y_train_test)

# time: 2:50:00

plt.scatter(X_train_temp, y_train_temp, label= "Data", color='blue')
x = tf.linspace(-20, 40, 100)
plt.plot(x, temp_reg.predict(np.array(x).reshape(-1, 1)), label= "Fit", color= 'red', linewidth= 3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Bike Count")
plt.xlabel("Temperature (C)")
plt.show()


# Multiple Linear Regression


train, val, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])


_, X_train_all, y_train_all  = get_xy(train, "bike_count", x_labels= df.columns[1:])
_, X_train_all, y_train_all  = get_xy(val,   "bike_count", x_labels= df.columns[1:])
_, X_train_all, y_train_all  = get_xy(test,  "bike_count", x_labels= df.columns[1:])

all_reg = LinearRegression()
all_reg.fit(X_train_all, y_train_all)

all_reg.score(X_train_all, y_train_all)

y_pred_lr = all_reg.predict(X_train_all)


# Reggression With Neural Networks

def plot_history(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.set_xlabel('Epoch')
    plt.set_ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()


temp_norm = tf.keras.layers.Normalization(input_shape= [1,], axis= None)
temp_norm.adapt(X_train_temp.reshape(-1))

temp_model = tf.keras.Sequential([
    temp_norm,
    tf.keras.layers.Dense(1)
    
])

temp_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= 0.01),
                       loss= 'mean_squared_error')

history = temp_model.fit(X_train_temp, y_train_temp, 
                         epochs= 1000, verbose=0, 
                         validation_data= (X_train_temp, y_train_temp))


# Neural Net

nn_model = tf.keras.Sequential([
    temp_norm,
    tf.keras.layers.Dense(32, activation= 'relu'),
    tf.keras.layers.Dense(32, activation= 'relu'),
    tf.keras.layers.Dense(32, activation= 'relu'),
    tf.keras.layers.Dense(1)
    
])

nn_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= 0.001),
                          loss= 'mean_squared_error')

history = nn_model.fit(X_train_temp, y_train_temp,
                       validation_data= (X_train_val, y_train_val),
                       verbose=0, epochs= 100
)

plot_history(history)

plt.scatter(X_train_temp, y_train_temp, label= "Data", color='blue')
x = tf.linspace(-20, 40, 100)
plt.plot(x, nn_model.predict(np.array(x).reshape(-1, 1)), label= "Fit", color= 'red', linewidth= 3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Bike Count")
plt.xlabel("Temperature (C)")
plt.show()

all_norm = tf.keras.layers.Normalization(input_shape= [6,], axis= 1)
all_norm.adapt(X_train_all)


nn_model = tf.keras.Sequential([
    all_norm,
    tf.keras.layers.Dense(32, activation= 'relu'),
    tf.keras.layers.Dense(32, activation= 'relu'),
    tf.keras.layers.Dense(32, activation= 'relu'),
    tf.keras.layers.Dense(1)
    
])

nn_model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= 0.001),
                          loss= 'mean_squared_error')


history = nn_model.fit(X_train_all, y_train_all,
                       validation_data= (X_train_val, y_train_val),
                       verbose= 0, epochs= 100
)

plot_history(history)


# calculate the MSE for both linear reg and nn
y_pred_lr = all_reg.predict(X_train_all)
y_pred_nn = nn_model.predict(X_train_all)


def MSE(y_pred, y_real):
    return (np.square(y_pred - y_real)).mean()


MSE(y_pred_lr, y_train_all)

MSE(y_pred_nn, y_test_all)



""" Dataset:    
    
    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
    [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
    School of Information and Computer Science.
    
    Source: Data Source: http://data.seoul.go.kr/ SOUTH KOREA PUBLIC HOLIDAYS. URL: publicholidays.go.kr """
    
