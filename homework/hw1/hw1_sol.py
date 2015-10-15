from __future__ import print_function
import cvxpy as cvx
import numpy as np

def prob1(A, b):
    m,n = A.shape

    # define variables ...
    x = cvx.Variable(n)
    
    prob = cvx.Problem(cvx.Minimize(0), [A*x == b, x >= 0])
    prob.solve()
    
    if prob.status == 'infeasible':
        return None
    elif prob.status == 'optimal':
        return np.array(x.value).flatten()
    else:
        print("Shouldn't happen for the data I'll give you.")

def prob2(X, a):
    n, m = X.shape
    
    t = cvx.Variable(m)
    prob = cvx.Problem(cvx.Minimize(0), [X*t == a, cvx.sum_entries(t) == 1, t >= 0])
    prob.solve()
    if prob.status == 'infeasible':
        return False
    else:
        return True


def prob3(A, b, X):
    n, m = X.shape
    
    t = cvx.Variable(m)
    a = cvx.Variable(n)

    prob = cvx.Problem(cvx.Minimize(0), [A*a <= b, X*t == a, cvx.sum_entries(t) == 1, t >= 0])
    prob.solve()
    
    if prob.status == 'infeasible':
        return False
    else:
        return True