% CME 252: Gradient Descent
% AJ Friend \
  ICME, Stanford University

# Introduction
## Introduction
- min f(x)
- assumptions!: smooth, differentiable
- find grad f x = 0
- assume min is attained, p star
- iterative method to produce poitns xk
- such that f(xk) geos to p star


## Examples
- Ax=b, least squares
- ||Ax-b||^2_2
- <Ax-b, ax-b> = x^TA^TAx + ...
- gradient to zero gives normal equations
- equivalent to x^TQx + ... for PSD Q
- gradient tricks for vectors and matrices. Matrix cookbook
- general f(Ax+b) gradient rule
- logsumexp(Ax+b) derive gradient. fits into logistic regression
- maybe even an exact logistic regresison example

## gradients
- intuition: points in direciton of steepest ascent
- tangent affine underestimator to convex function

## prototype alg:
- xk+1 = xk - alphak gk
- repeat
stop when grad f == 0 (approximately)


## stopping criteria
- norm(grad) < eps
- f(x ) - pstar < eps (if known)

## Grad descent example
- fixed step size, plot function values
- plot norm of gradient
- to big, see oscillation
- too small, slow convergence
- just right, fast
- how to choose?

## curvature
- strongly convex means minimum eigenvalue
- lipschits continuous means max eigenvalue of hessian
- need to explain taylor series

## quadratic under and over estimators
- convergence analysis

## line search
- probably not too much
- simple linesearch based on quadratics

## Stochastic gradient descent
- and in parallel
- decreasing stepsize (show example of hopping)
- give an example stepsize but skip analysis

## Nonconvex problems
- restarting
- monitoring
- babysitting

## Newton's method
- better quadratic approximation
- but what if not PSD....

## Problems with Constraints
- simple barrier methods
