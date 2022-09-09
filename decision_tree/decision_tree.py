from typing import Counter
import numpy as np 
import pandas as pd 
from copy import deepcopy
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)



class Node:
    def __init__(self):
        self.children = {} # dictionary, with key = specific value of the node attribute
        self.parent = None
        self.attribute = None
        self.label = None


class DecisionTree:
    
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.children = {} # dictionary, with key = specific value of the node attribute
        self.parent = None
        self.attribute = None
        self.label = None
        self.visited_children = 0
    
    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """

        #self.label = y.mode()[0]

        

        attrs = X.columns #attributes are columns in the DataFrame object
        most_common_target = y.mode()[0]
        #node = Node()
        # If all examples are positive (or all negative), return the single Node tree root with positive/negative label
        # In other words, if all y have the same value, the Node gets the label equal to that value
        if len(np.unique(y))==1 :      
            self.label = np.unique(y)[0]  
        # If attributes is empty, return the root, with label = most common target attribute
        elif len(attrs) == 0:
            
            self.label = most_common_target
        else:
            # A = attr from attrs that best classifies Examples (y-s)
            A = max(attrs, key = lambda attr: gain(X, y, attr))
            # the decision attr for root = A
            self.attribute = A
            possible_vals = X[A].unique()  

            ### task b related
            child_node = DecisionTree()
            child_node.parent = self
            child_node.label = most_common_target
            self.children["other"] = child_node

            ###

            for val in possible_vals:
                # find subset of X, y where A = val
                y_subset = y[X[A] == val]
                X_subset = X[X[A] == val]      
                child_node = DecisionTree()
                child_node.parent = self
                if (len(y_subset) == 0):
                    child_node.label = y.mode()[0]
                    self.children[val] = child_node
                else:    
                    child_node.fit(X_subset.drop(columns =[A]), y_subset)
                    self.children[val] = child_node





    
    
    

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        y = list()
        for i in range(X.index[0], X.index[-1]+1):
            row = X.loc[i]
            dt = self
            attr = dt.attribute
            while (attr is not None):
                if row[attr] in dt.children.keys():
                    dt = dt.children[row[attr]]
                    attr = dt.attribute
                    label = dt.label
                else:
                    label = dt.children["other"].label
                    attr = None
            y.append(label)
            dt = go_to_root(dt)
            

        #y = pd.DataFrame(y)
        return np.array(y)

    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        visited = list() # visited nodes
        tuples = list()
    
        matrix = list()
        traverse_tree(self, [], matrix)
        return matrix


    

# --- Some utility functions 
def traverse_tree(dt, tuples, matrix, visited = list()):
    if len(dt.children) == dt.visited_children:
        visited.append(dt)
    if dt.label is not None:
        rule = list()
        rule.append(tuples)
        rule.append(dt.label)
        go_to_root(dt)
        matrix.append(rule)
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
            
def gain(X, y, A):
        """
        returns the gain of an attribute A relative to the collection with examples X and corresponding targets y 
        """
        values = X[A].unique()
        y_partitioned = [y[X[A] == val] for val in values]
        val_counts = y.value_counts()
        return  entropy(val_counts) - np.sum([entropy(partition.value_counts())*len(partition)/len(y) for partition in y_partitioned]) # according (3.4), page 58

def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))


def go_to_root(dt):
    while (dt.parent is not None):
        dt = dt.parent
    return dt