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

    y = dataframe[y_label]
    data =
        



_, X_train, y_train = 

""" Dataset:    
    
    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
    [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
    School of Information and Computer Science.
    
    Source: Data Source: http://data.seoul.go.kr/ SOUTH KOREA PUBLIC HOLIDAYS. URL: publicholidays.go.kr """
    

