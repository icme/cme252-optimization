from __future__ import print_function
import cvxpy as cvx
import numpy as np

"""
Please fill in your name and email address in the variables below.

Please complete prob1, prob2, and prob3 according to the description in hw1.pdf.
"""

# modify these variables
name = YOUR NAME
stanford_email = yourname@stanford.edu

def prob1(A, b):
    m,n = A.shape

    # define variables
    x = ...
    # create problem
    # solve problem
    
    prob = ....
    
    if prob.status == 'infeasible':
        return None
    elif prob.status == 'optimal':
        # convert x.value into an numpy array, and flatten it to 1D
        return np.array(x.value).flatten()
    else:
        print("Shouldn't happen for the data I'll give you.")

def prob2(X, a):
    n, m = X.shape
    # fill in code

def prob3(A, b, X):
    n, m = X.shape
    
    # fill in code