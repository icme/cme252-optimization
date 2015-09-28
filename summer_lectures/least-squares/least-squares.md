---
title: Linear Least Squares
author: |
  | Nick Henderson, AJ Friend
  | Stanford University
date: August 19, 2015
---

# Linear least squares

## Linear least squares overview

- Ubiquitous statistical model
- Applications everywhere
- Goal: find linear model $f(x)$ that fits your data
- Useful model for the purpose of studying optimization

## Let's start with data

\begin{center}
\begin{tabular}{cc}
$x$: independent variable & $y$: response variable \\
\hline
0.0 & 0.46 \\
0.11 & 0.31 \\
0.22 & 0.38 \\
0.33 & 0.39 \\
0.44 & 0.65 \\
0.56 & 0.4 \\
0.67 & 0.87 \\
0.78 & 0.69 \\
0.89 & 0.87 \\
1.0 & 0.88 \\
\end{tabular}
\end{center}

## Where might this data come from?

\begin{center}
\begin{tabular}{cc}
$x$: independent variable & $y$: response variable \\
\hline
height              & weight \\
square feet         & price of home \\
device property     & failure rate \\
stock market return & individual asset return
\end{tabular}
\end{center}

\note{
- stock market example is known as beta
- beta may be derived from capital asset pricing model (CAPM)
}

## Where did this data come from?

A linear model with random error:

$$
y_i = m \cdot x_i + b + \sigma \cdot \epsilon_i
$$

The error is standard normal:

$$
\epsilon_i \sim N(0,1)
$$

Code to generate in python:

```
np.random.seed(1)
m = 0.6
b = 0.3
sigma = .1
x = np.linspace(0,1,10)
y = m*x + b + sigma*np.random.standard_normal(x.shape)
```

## Let's plot the data

\centering
\includegraphics[width=0.7\textwidth]{fig/1d-fit-1.pdf}

## Let's draw a line through it

\centering
\includegraphics[width=0.7\textwidth]{fig/1d-fit-2.pdf}

## Why do we want to do this?

- We have data. In the previous figures we show 2-dimensional data with points $(x_i,y_i)$
- Want to better understand data
- Want to use data for useful things, for example to make predictions
- We can do both by building a model
$$
y \approx f(x)
$$
- Model has parameters. We use optimization to compute those parameters.

## Linear models in one dimension

The model is:
$$
y \approx m \cdot x + b
$$

- $x$ is the independent variable
- $y$ is the dependent or response variable
- $m$ is the slope
- $b$ is the $y$-intercept
- This is linear regression

## Fitting the model to data

- Any given data point is going to result in some model error
- In optimization and linear algebra, we call this error the *residual*
- Given $n$ data points $(x_i,y_i$), the residual from the linear model is
simply
$$ r_i = m\cdot x_i + b - y_i $$
- To fit the model to data, we set up an optimization problem that chooses
parameters $m$ and $b$ to minimize the sum of squared residuals:

$$
\begin{array}{ll}
\mbox{minimize} & \frac{1}{2} \sum_{i=1}^n r_i^2
= \frac{1}{2} \sum_{i=1}^n (m\cdot x_i + b - y_i)^2
\end{array}
$$

- If the errors are distributed according to the normal distribution, then the
  solution to this optimization problem maximizes the log-likelihood of the model.

## Fitting the model to data

\centering
\includegraphics[width=0.7\textwidth]{fig/1d-fit-3.pdf}

## Let's optimize by hand

- Don't worry, we won't do very much of this
- One way to write the model is
$$
y = f(x;m,b)
$$
- The semi-colon separates the independent variable $x$ from the parameters
  $(m,b)$
- We can write the objective function as $\phi(m,b)$
$$
\begin{array}{ll}
\mbox{minimize} & \phi(m,b) = \frac{1}{2} \sum_{i=1}^n (f(x_i;m,b) - y_i)^2
\end{array}
$$
- To solve the optimization problem we find parameters $m$ and $b$ such that
  $\nabla \phi(m,b) = 0$

## Partial derivatives & system of equations

Partial derivatives:

$$
\frac{\partial \phi}{\partial m}
  = \frac{1}{2} \sum_{i=1}^n \frac{\partial}{\partial m} (m\cdot x_i + b - y_i)^2
  = m \sum x_i^2 + b \sum x_i - \sum x_i y_i
$$

$$
\frac{\partial \phi}{\partial b}
  = \frac{1}{2} \sum_{i=1}^n \frac{\partial}{\partial b} (m\cdot x_i + b - y_i)^2
  = m \sum x_i + nb - \sum y_i
$$

System of equations:

$$
\frac{\partial \phi}{\partial m} = 0
$$

$$
\frac{\partial \phi}{\partial b} = 0
$$

## Solution

$$
m = \frac{\sum x_i y_i - \frac{1}{n}\sum x_i \sum y_i}
         {\sum x_i^2 - \frac{1}{n}(\sum x_i)^2}
$$

$$
b = \frac{\sum y_i - m\sum x_i}
         {n}
$$

## Let's look at $m$

Something looks nice here:

$$
m = \frac{\sum x_i y_i - \frac{1}{n}\sum x_i \sum y_i}
         {\sum x_i^2 - \frac{1}{n}(\sum x_i)^2}
$$

Multiply both numerator and denominator by $1/n$:

$$
m = \frac{\frac{1}{n} \sum x_i y_i - \frac{1}{n}\sum x_i \frac{1}{n}\sum y_i}
         {\frac{1}{n} \sum x_i^2 - (\frac{1}{n} \sum x_i)^2}
$$

We see sample covariance and variance here!

$$
m = \frac{\text{cov}(X,Y)}
         {\text{var}(X)}
$$

## Let's solve in python!

Code:
```
# solve via numpy covariance function
A = np.vstack((x,y))
V = np.cov(A)
m_est = V[0,1] / V[0,0]
b_est = (y.sum() - m_est*x.sum()) / len(x)
print(m_est)
print(b_est)
```
Result:
```
m_est = 0.566036432757  (true value = 0.6)
b_est = 0.307267694541  (true value = 0.3)
```

## Look at the plot

\centering
\includegraphics[width=0.7\textwidth]{fig/1d-fit-4.pdf}

## Solve in CVXPY

Remember the optimization problem: $\text{minimize } \frac{1}{2} \sum_{i=1}^n (m\cdot x_i + b - y_i)^2$

We can write this almost directly in python:

```
from cvxpy import *
# Construct the problem.
m_cvx = Variable()
b_cvx = Variable()
objective = Minimize(sum_squares(m_cvx*x + b_cvx - y))
prob = Problem(objective)
# The optimal objective is returned by prob.solve().
result = prob.solve()
```

```
m_cvx.value = 0.56604, b_cvx.value = 0.30727
```

# Linearization

## Why linear?

Recall from calculus the Taylor series expansion for an infinitely
differentiable function $f$ expanded about point $a$:

$$
f(x-a) = f(a) + \frac{1}{1!}f'(a)(x-a) + \frac{1}{2!}f''(a)(x-a)^2 + \frac{1}{3!}f'''(a)(x-a)^3 + \cdots
$$

## Linear expansion

\centering
\includegraphics[width=\textwidth]{fig/taylor-series.pdf}

## Linear model

Taylor series expansion:

$$
f(x-a) = f(a) + \frac{1}{1!}f'(a)(x-a) + \frac{1}{2!}f''(a)(x-a)^2 + \frac{1}{3!}f'''(a)(x-a)^3 + \cdots
$$

Linear model (with errors):

$$
y = b + m\cdot x + \epsilon
$$

# Non-linear data

## What about this data?

\centering
\includegraphics[width=0.7\textwidth]{fig/exp-fit-1.pdf}

## Let's fit a linear model

\centering
\includegraphics[width=0.7\textwidth]{fig/exp-fit-2.pdf}

## We can fit an exponential model

The model:

$$
y \approx m \cdot e^x + b
$$

Note: this model is still linear in the parameters $m$ and $b$!  We just need to
transform the independent variable and then solve using the same technique.  In
code with CVXPY:

```
m = Variable()
b = Variable()
objective = Minimize(sum_squares(m*np.exp(x) + b - y))
prob = Problem(objective)
result = prob.solve()
```

## Result

\centering
\includegraphics[width=0.7\textwidth]{fig/exp-fit-3.pdf}

# Least squares in matrix-vector form

## More variables

- We've been talking about least squares with one independent variable
- Utility of linear models is increased when we incorporate more variables
- For this discussion, we need vectors and matrices!

## Vectors

- A *vector* is an array of numbers
- Vectors have a length or number of elements, usually denoted $n$
- By default we think of vectors as numbers being stacked in a column

$$
x = \begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
\vdots \\
x_n
\end{bmatrix}
$$

## Vector operations

- Multiplication by scalars

$$
\alpha x = \begin{bmatrix}
\alpha x_1 \\
\alpha x_2 \\
\vdots \\
\alpha x_n
\end{bmatrix}
$$

- Vector addition

$$
x + y = \begin{bmatrix}
x_1 + y_1 \\
x_2 + y_2\\
\vdots \\
x_n + y_n
\end{bmatrix}
$$

## Column and row vectors

By convention, we consider vectors to be oriented as columns:

$$
x = \begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
\vdots \\
x_n
\end{bmatrix}
$$

Sometimes we want a row vector.  We use the transpose operation for this:

$$
x^T = \begin{bmatrix}
x_1 \ 
x_2 \ 
x_3 \ 
\cdots \ 
x_n
\end{bmatrix}
$$

## Vector inner product

- $x$ and $y$ are both vectors with length $n$
- Also known as a dot product

$$
x^T y = x \cdot y = \sum_{i=1}^n x_i y_i
$$

- Inner product between $x$ and $x$

$$
x^T x = x \cdot x = \sum_{i=1}^n x_i^2
$$

## Vector norm

\centering
\includegraphics[width=\textwidth]{fig/vector-norm.pdf}

## Vector norm

- Vectors have a magnitude or norm
- The most widely used norm is the 2-norm or sum of squares of the elements:
$$
||x||_2 = \sqrt{\sum_{i=1}^n (x_i)^2  } = \sqrt{x^T x}
$$
- The 2-norm and is sometimes written without the 2 as $||x||$
- There is also a 1-norm, which is the sum of the absolute values of the vector elements:
$$
||x||_1 = \sum_{i=1}^n |x_i|
$$

## Least squares objective

- Let's say we have a data set $(a_i,y_i)$
- $a$ is now the independent variable and $y$ is the response variable
- The linear model: $y = m\cdot a + b + r$
- The optimization formulation to find parameters $m$ and $b$ is

$$
\begin{array}{ll}
\mbox{minimize} & \frac{1}{2} \sum_{i=1}^n (m\cdot a_i + b - y_i)^2
= \frac{1}{2} || m\cdot a + b - y ||_2^2
= \frac{1}{2} ||r||_2^2
\end{array}
$$

## Matrices

- A *matrix* is an array of numbers with height $m$ and width $n$
- We write matrix elements with lower case letters and indexed by row then
  column
    - $a_{ij}$ refers to element in row $i$ and column $j$

$$
A = \begin{pmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{33} \\
a_{31} & a_{32} & a_{33}
\end{pmatrix}
$$

## Matrix-vector products

- Say $A$ is an $m\times n$ matrix and $x$ is an $n$-vector
- Length of vector must match the second dimension of matrix
- The product is: $y = Ax$
- $y$ is an $m$-vector, the number of rows in $A$
- Formula for the product: $y_i = \sum_{j=1}^n a_{ij}x_j$
- Example:

$$
y = Ax =
\begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22} \\
a_{31} & a_{32}
\end{pmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
= 
\begin{bmatrix}
a_{11}x_1 + a_{12}x_2 \\
a_{21}x_1 + a_{22}x_2 \\
a_{31}x_1 + a_{32}x_2
\end{bmatrix}
$$

## Matrix-matrix product

- $A$ is an $m\times n$ matrix
- $B$ is an $n\times p$ matrix
- $C = AB$ is the product of $A$ and $B$ and is an $m\times p$ matrix
- The elements of $C$ may be computed with

$$
c_{ij} = \sum_{k=1}^n a_{ik} b_{kj}
$$

- Size of the inner dimensions of the matrices must match

## Matrix-vector product revisited

$$
y = Ax =
\begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22} \\
a_{31} & a_{32}
\end{pmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
= 
\begin{bmatrix}
a_{11}x_1 + a_{12}x_2 \\
a_{21}x_1 + a_{22}x_2 \\
a_{31}x_1 + a_{32}x_2
\end{bmatrix}
$$

- we can consider the column vector to be a $n\times 1$ matrix
- matrix-vector product is special case of matrix-matrix product
- the inner product between vectors is a product between matrices of size
  $1\times n$ and $n\times 1$

$$
x^T y = \sum_{i=1}^n x_i y_i
$$

## Matrix-vector form for least squares

$$
\begin{array}{ll}
\mbox{minimize} & \frac{1}{2} \sum_{i=1}^n (m\cdot a_i + b - y_i)^2
\end{array}
$$

- Pack data from the independent variable and a constant into matrix $A$ and
  model parameters into vector $x$:

$$
A = \begin{pmatrix}
a_1 & 1 \\
a_2 & 1 \\
\vdots & \vdots \\
a_n & 1
\end{pmatrix},
\hspace{.2in}
x = \begin{bmatrix}
m \\
b
\end{bmatrix}
$$

- Linear model for the data is now:

$$
y = Ax + r
$$

## Standard form for least squares

$$
\begin{array}{ll}
\mbox{minimize} & \frac{1}{2} ||Ax - b||_2^2
\end{array}
$$

In the context of model fitting:

- $A$ is a matrix that contains data from independent variables
- $b$ is the vector holding response data
- $x$ is the vector of model parameters
- The optimization problem above is solved to find $x^*$, the parameters that
  minimize the sum of squared residuals
- For each item of data, we have the equation where $a_i^T$ is row $i$ of $A$

$$
a_i^T x - b_i = r_i
$$

## Notation from statistics

$$
\begin{array}{ll}
\mbox{minimize} & \frac{1}{2} ||\mathbf{y} - \mathbf{X}\beta||_2^2
\end{array}
$$

The statistics community often uses different notation:

- $\mathbf{X}$ is the matrix of input data
- $\mathbf{y}$ is the vector of response data
- $\beta$ is the vector of model parameters

## CVXPY for least squares

```
np.random.seed(1); n = 10 # number of data points
input_data = np.linspace(0,1,n)
response_data = 0.6*input_data + 0.3 + 0.1*np.random.standard_normal(n)
# problem data
A = np.vstack([input_data,np.ones(n)]).T; b = response_data
# cvx problem
x = Variable(A.shape[1])
objective = Minimize(sum_squares(A*x - b))
prob = Problem(objective); result = prob.solve()
# get value & print
x_star = np.array(x.value)
print('slope = {:.4}, intercept = {:.4}'.format(x_star[0,0],x_star[1,0]))
```

```
slope = 0.566, intercept = 0.3073
```

# Examples

## What about this data?

\centering
\includegraphics[width=0.7\textwidth]{fig/poly-fit-1.pdf}

## We can try a polynomial model

- Have $m$ data points $(u_i,y_i)$
- The model

$$
y \approx p(u) = x_1 + x_2 u + x_3 u^2 + \cdots + x_n u^{n-1}
$$

- $u$ is the independent variable
- $y$ is the response variable
- $x_i$ are the model parameters and coefficients of the polynomial
- Model is linear in the parameters!  We can use least squares!

## Linear model

$$
\begin{bmatrix}
y_1 \\
y_2 \\
y_3 \\
y_4 \\
\vdots \\
y_m
\end{bmatrix}
\approx
\begin{bmatrix}
1 & u_1 & u_1^2 & \dots & u_1^{n-1}\\
1 & u_2 & u_2^2 & \dots & u_2^{n-1}\\
1 & u_3 & u_3^2 & \dots & u_3^{n-1}\\
1 & u_4 & u_4^2 & \dots & u_4^{n-1}\\
\vdots & \vdots & \vdots & \ddots &\vdots \\
1 & u_m & u_m^2 & \dots & u_m^{n-1}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
\vdots \\
x_n
\end{bmatrix}
$$

$$
y \approx Ax
$$

## Solve with CVXPY

```
def cvxpy_poly_fit(x,y,degree):
    # construct data matrix
    A = np.vander(x,degree+1)
    b = y
    p_cvx = Variable(degree+1)
    # set up optimization problem
    objective = Minimize(sum_squares(A*p_cvx - b))
    constraints = []
    # solve the problem
    prob = Problem(objective,constraints)
    prob.solve()
    # return the polynomial coefficients
    return np.array(p_cvx.value)
```

## Linear fit

\centering
\includegraphics[width=0.7\textwidth]{fig/poly-fit-2.pdf}

## Quadratic fit

\centering
\includegraphics[width=0.7\textwidth]{fig/poly-fit-3.pdf}

## Cubic fit

\centering
\includegraphics[width=0.7\textwidth]{fig/poly-fit-4.pdf}

## Generating model

\centering
\includegraphics[width=0.7\textwidth]{fig/poly-fit-5.pdf}

## Example: time series smoothing

- noisy observations at regular interval, $y \in \mathbf{R}^n$ (discritized curve)
- don't have a model for the curve (linear, polynomial, ...)
- do know the the curve should be "smooth"
- idea: find $x \in \mathbf{R}^n$ which is close to $y$, but also penalized for
  being nonsmooth

## Time series data

\centering
\includegraphics[width=0.7\textwidth]{fig/smooth-1.pdf}

## Optimization problem

- Want each $x_i$ to be close to each $y_i$
- Want vector $x$ to represent a smooth function
- Optimization problem

$$
\mbox{minimize} \ ||x-y||_2^2 + \rho \cdot \text{penalty}(x)
$$

- Use penalty function to encourage smoothness
- Use parameter $\rho$ to trade-off fit to data and smoothness

## Measurement of smoothness

- roughly define smoothness to be a curve which does not change slope much
- change in slope (of a smooth curve) given by the second derivative
- change in slope of a discretized curve given by the second-order differences,
  $Dx$, where

$$
D = \begin{pmatrix}
1 & -2 & 1 & 0 & \ldots & &&0 \\
0 & 1  & -2 & 1 & 0 & \ldots & &0\\
0 & 0 &1 & -2 & -1 & 0 & \ldots & 0 \\
\vdots &
\end{pmatrix}
$$

## Least squares model

- $x$ close to the data if $\|x-y\|_2^2$ is small
- $x$ is smooth if $\|Dx\|_2^2$ is small
- solve the least-squares problem
$$
\begin{array}{ll}
\mbox{minimize} & \|x-y\|_2^2 + \rho \|Dx\|_2^2
\end{array}
$$
- $\rho$ trades-off between fidelity to the data and smoothness

## Standard form

Model:

$$
\begin{array}{ll}
\mbox{minimize} & \|x-y\|_2^2 + \rho \|Dx\|_2^2
\end{array}
$$

Standard form:

$$
\mbox{minimize}\  \left\lVert
\begin{pmatrix}
I \\
\rho D
\end{pmatrix}
x -
\begin{pmatrix}
y \\
0
\end{pmatrix}
\right\rVert_2^2
$$

## Solve the problem in CVXPY

```
# get second-order difference matrix
D = diff(n, 2)
rho = 1
# construct and solve problem
x = cvx.Variable(n)
cvx.Problem(cvx.Minimize(cvx.sum_squares(x-y)
                         +rho*cvx.sum_squares(D*x))).solve()
x = np.array(x.value).flatten()
```

## $\rho = 1$

\centering
\includegraphics[width=0.7\textwidth]{fig/smooth-2.pdf}

## $\rho = 10$

\centering
\includegraphics[width=0.7\textwidth]{fig/smooth-3.pdf}

## $\rho = 1000$

\centering
\includegraphics[width=0.7\textwidth]{fig/smooth-4.pdf}
