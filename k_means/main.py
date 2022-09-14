from copy import deepcopy
from enum import unique
from ntpath import join
from tkinter import Y
import numpy as np 
import pandas as pd 
import k_means as km
            

def main():
    #data_1 = pd.read_csv('data_1.csv')
    #X = data_1[['x0', 'x1']]
    #model_1 = km.KMeans()
    #model_1.fit(X)
    #z = model_1.predict(X)
    #print(z)

    data_2 = pd.read_csv('data_2.csv')
    # Fit Model 
    X = data_2[['x0', 'x1']]
    model_2 = km.KMeans(10)  # <-- Feel free to add hyperparameters 
    model_2.fit(X)

    #z = model_2.predict(X)


if __name__ == '__main__':
     main()