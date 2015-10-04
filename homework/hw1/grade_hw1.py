from __future__ import print_function
import numpy as np

# import your hw1.py solution
import hw1

# numerical tolerance for tests of equivalence
tol = 1e-5

def test_prob1():
    print('Testing Problem 1')
    passed = True

    # This data should be infeasible
    A = np.array([[ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799],
            [-0.97727788,  0.95008842, -0.15135721, -0.10321885,  0.4105985 ],
            [ 0.14404357,  1.45427351,  0.76103773,  0.12167502,  0.44386323]])

    b = np.array([ 0.33367433,  1.49407907, -0.20515826])

    x = hw1.prob1(A, b)
    if x is not None:
        passed = False
        print('Fail 1')

    # This data should be feasible
    A = np.array([[ 1.62434536, -0.61175641, -0.52817175, -1.07296862,  0.86540763],
            [-2.3015387 ,  1.74481176, -0.7612069 ,  0.3190391 , -0.24937038],
            [ 1.46210794, -2.06014071, -0.3224172 , -0.38405435,  1.13376944]])
    b = np.array([-1.09989127, -0.17242821, -0.87785842])

    x = hw1.prob1(A, b)
    if np.linalg.norm(A.dot(x) - b) <= tol and np.all(x >= -tol):
        # then A*x == b and x >= 0 (up to numerical tolerance)
        pass
    else:
        passed = False
        print('Fail 2')

    return passed

def test_prob2():
    print('Testing Problem 2')

    passed = True
    X = np.array([[-0.27909772,  1.62284909,  0.01335268, -0.6946936 ,  0.6218035 ,
             -0.59980453,  1.12341216,  0.30526704,  1.3887794 , -0.66134424],
            [ 3.03085711,  0.82458463,  0.65458015, -0.05118845, -0.72559712,
             -0.86776868, -0.13597733, -0.79726979,  0.28267571, -0.82609743],
            [ 0.6210827 ,  0.9561217 , -0.70584051,  1.19268607, -0.23794194,
              1.15528789,  0.43816635,  1.12232832, -0.9970198 , -0.10679399],
            [ 1.45142926, -0.61803685, -2.03720123, -1.94258918, -2.50644065,
             -2.11416392, -0.41163916,  1.27852808, -0.44222928,  0.32352735],
            [-0.10999149,  0.00854895, -0.16819884, -0.17418034,  0.4611641 ,
             -1.17598267,  1.01012718,  0.92001793, -0.19505734,  0.80539342]])
    a = np.array([ 0.43051423,  0.07223898,  0.2836541 , -0.67100498,  0.17019935])

    # a should be inside the convex hull
    if hw1.prob2(X, a) is True:
        print('Passed 1')
    else:
        print('Failed 1')
        passed = False

    # -a should not be inside the convex hull
    if hw1.prob2(X, -a) is False:
        print('Passed 2')
    else:
        print('Failed 2')
        passed = False

    return passed

def test_prob3():
    print('Testing Problem 3')

    passed = True

    A = np.array([[ 0.5488135 ,  0.71518937,  0.60276338],
            [ 0.54488318,  0.4236548 ,  0.64589411],
            [ 0.43758721,  0.891773  ,  0.96366276],
            [ 0.38344152,  0.79172504,  0.52889492],
            [-1.        , -0.        , -0.        ],
            [-0.        , -1.        , -0.        ],
            [-0.        , -0.        , -1.        ]])
    b = np.array([ 0.56804456,  0.92559664,  0.07103606,  0.0871293 ,  0.        ,
             0.        ,  0.        ])

    X = np.array([[1,1,1],[1,0,0],[0,1,0]]).T

    if hw1.prob3(A, b, X) is False:
        print('Passed 1')
    else:
        print('Failed 1')
        passed = False

    X = np.array([[.01,.01,.01],[1,0,0],[0,1,0]]).T

    if hw1.prob3(A, b, X) is True:
        print('Passed 2')
    else:
        print('Failed 2')
        passed = False

    return passed

points = 0
total = 3

for f in test_prob1, test_prob2, test_prob3:
    if f():
        points += 1

print("Got {} out of {} points!".format(points, total))



