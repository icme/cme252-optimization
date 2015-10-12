import numpy as np
import cvxpy as cvx

"""
Please fill in your name and email address in the variables below.

Please complete prob1, prob2, and prob3 according to the description in hw2.pdf.
"""

# modify these variables
name = "YOUR NAME"
stanford_email = "yourname@stanford.edu"

def prob1():
    answer = {}
    answer[1] = None
    answer[2] = None
    answer[3] = None
    answer[4] = None
    answer[5] = None
    answer[6] = None

    return answer


def prob2(x,y):
    n = len(x)
    m = 4

    w = cvx.Variable(m)

    # fill in code

    return np.array(w.value).flatten()

def prob3(x,y):
    n = len(x)
    m = 4

    w = cvx.Variable(m)

    # fill in code
    
    return np.array(w.value).flatten()