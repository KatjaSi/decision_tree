from copy import deepcopy
from enum import unique
from ntpath import join
from tkinter import Y
import numpy as np 
import pandas as pd 
import decision_tree as dt

from decision_tree import  DecisionTree, Node, go_to_root

def main():
    """data_1 = pd.read_csv('data_2.csv')
    X = data_1.drop(columns=['Play Tennis'])
    y = data_1['Play Tennis']
    #print(y.shape)
    dt = DecisionTree()
    dt.fit(X,y)
    #dt.fit(X,y)
    #for row in X.iterrows():
     #   print(row)

    rules = dt.get_rules()
    print(rules)
"""
    data_2 = pd.read_csv('data_2.csv')
    data_2_train = data_2.query('Split == "train"')
    data_2_valid = data_2.query('Split == "valid"')
    data_2_test = data_2.query('Split == "test"')
    X_train, y_train = data_2_train.drop(columns=['Outcome', 'Split']), data_2_train.Outcome
    X_valid, y_valid = data_2_valid.drop(columns=['Outcome', 'Split']), data_2_valid.Outcome
    X_test, y_test = data_2_test.drop(columns=['Outcome', 'Split']), data_2_test.Outcome
    data_2.Split.value_counts()


    model_2 = dt.DecisionTree()  # <-- Feel free to add hyperparameters 
    model_2.fit(X_train, y_train)

    for rules, label in model_2.get_rules():
        conjunction = ' ∩ '.join(f'{attr}={value}' for attr, value in rules)
        print(f'{"✅" if label == "success" else "❌"} {conjunction} => {label}')

    print(f'Train: {dt.accuracy(y_train, model_2.predict(X_train)) * 100 :.1f}%')
    print(f'Valid: {dt.accuracy(y_valid, model_2.predict(X_valid)) * 100 :.1f}%')


   

    
    
"""
    [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]

        """
"""
    data_2 = pd.read_csv('data_2.csv')
    data_2_train = data_2.query('Split == "train"')
    data_2_valid = data_2.query('Split == "valid"')
    data_2_test = data_2.query('Split == "test"')
    X_train, y_train = data_2_train.drop(columns=['Outcome', 'Split']), data_2_train.Outcome
    X_valid, y_valid = data_2_valid.drop(columns=['Outcome', 'Split']), data_2_valid.Outcome
    X_test, y_test = data_2_test.drop(columns=['Outcome', 'Split']), data_2_test.Outcome



    model_2 = dt.DecisionTree()  # <-- Feel free to add hyperparameters 
    model_2.fit(X_train, y_train)
    #print(f'Train: {dt.accuracy(y_train, model_2.predict(X_train)) * 100 :.1f}%')
    #print(X_train.columns)
    #print(X_valid.columns)

    #y = model_2.predict(X_train)
    print(X_train)
    print("----")
    print(X_valid)
    #print(f'Valid: {dt.accuracy(y_valid, model_2.predict(X_valid)) * 100 :.1f}%')
    """


if __name__ == '__main__':
     # execute only if run as a script
     main()