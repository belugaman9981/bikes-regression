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
    


""" Dataset:    
    
    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
    [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
    School of Information and Computer Science.
    
    Source: Data Source: http://data.seoul.go.kr/ SOUTH KOREA PUBLIC HOLIDAYS. URL: publicholidays.go.kr """
    

