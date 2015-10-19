# notes for pandoc/markdown/beamer slides
links with various tricks to get things working.

# links
https://stackoverflow.com/questions/29532214/add-author-affiliation-in-r-markdown-beamer-presentation

http://andrewgoldstone.com/blog/2014/12/24/slides/

https://github.com/agoldst/tex/tree/master/lecture-slides

http://pandoc.org/demo/example9/producing-slide-shows-with-pandoc.html

https://github.com/sebastianbarfort/numedig/blob/master/2-workflow/slides/slides.md

look up "metadata blocks" in <http://pandoc.org/demo/example9/pandocs-markdown.html>

# todo
- maybe open class with regression to see CVXPY examples
    - students won't know convexity at this point
    - but this gives us more examples to work with when
      explaining convex functions, sets, problems in lecture 2
- with regression, can explain multiobjective problems
- give examples with convex problem transformations
- clean up SVM examples so less code is duplicated
- but each problem into a function of its own: SVM, SpVC, SVC..
- make it more explicit that convex <= concave gives a convex set
- add component-wise composition rule to make more interesting examples


## nonlinear discrimination
- adding features
- polynomial discrimination any different?
- rbf kernel? radial basis function
- kernel methods and relationship with convex opt...

## algorithms
- note that so far, we have said **nothing** about **how** to compute a supporting vector
- we have focused on modeling
- that's OK, we're focusing on modeling
- algorithms involve duality and optimality conditions

## scikitlearn comparison
- make sure it matches up with python SVM formulation
- maybe even do a timing comparison...

## data science perspective
- cleaning and centering data
- sparse predictors

# lecture ideas
- multiclass SVM
- max flow, multicommodity flow
- optimal evacuation plan

# homework
- given list of spheres (point in R^n and radius), find a point in the intersection
- find a point in intersection furthest in x direction
- intersection of "diamonds" l1-ball?
- ooh: list of norm balls: center, norm, raidus. find intersection
- polytopes: (need to introduce convex hull of points) given vertices of 2 convex polyhedra, determine if they intersect.
- hyperplane version of polytope?

## diet problem
- list of foods with nutrients
- min and max healthy range for each nutrient
- find a healthy diet

# Sun Oct 18 23:26:44 2015
- NMF
- min max rps
- max flow
- portfolio opt
- opt parade route
- evacuation
- structural opt
- rockets/trajectories
- control
- model predictive control

rps
max flow
parade route
opt evac
nmf
portfolio
