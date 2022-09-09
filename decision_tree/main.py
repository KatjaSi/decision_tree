from copy import deepcopy
from enum import unique
from ntpath import join
from tkinter import Y
import numpy as np 
import pandas as pd 
import decision_tree as dt

from decision_tree import  DecisionTree, Node, go_to_root

visited = list() # visited nodes
def traverse_tree(dt, tuples, matrix, visited = list()):
    if len(dt.children) == dt.visited_children:
        visited.append(dt)
    if dt.label is not None:
        tuples.append(dt.label)
        go_to_root(dt)
        matrix.append(tuples)
        visited.append(dt)
        tuples = list()
    else:
        if (dt in visited):
            if (dt.parent is not None):
                dt.parent.visited_children += 1
                return
        for key in dt.children.keys():
            tuples_copy = deepcopy(tuples)
            tuples_copy.append([dt.attribute, key])
            traverse_tree(dt.children[key], tuples_copy, matrix, visited)
            

def main():
    data_1 = pd.read_csv('data_1.csv')
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