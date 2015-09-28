# Installing Python and CVXPY

Overall requirements:
- Python 2.7
- CVXPY

## Mac OS X

- Download Anaconda python from <http://continuum.io/downloads>
- Follow the instructions here <http://docs.continuum.io/anaconda/install#mac-install>

- Check the terminal, has Anaconda python been added to the path?

- now from the terminal

- `conda update conda`, answer yes
- `conda update --all`, answer yes
- `pip install cvxpy`

- From the terminal, run the command `ipython notebook`
- Run the following cvxpy code from the notebook:

```
from cvxpy import *
import numpy

# Problem data.
m = 30
n = 20
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

# Construct the problem.
x = Variable(n)
objective = Minimize(sum_squares(A*x - b))
constraints = [0 <= x, x <= 1]
prob = Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
result = prob.solve()
# The optimal value for x is stored in x.value.
print x.value
# The optimal Lagrange multiplier for a constraint
# is stored in constraint.dual_value.
print constraints[0].dual_value
```

- You should see some vectors as output.

## Windows

??

## Linux

??
