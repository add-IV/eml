## Lecture 1

### Introduction

- current research
- ml applications
- mobile processor capability can't run modern ai
- HW decides which ideas succeed: Hardware Lottery
- organization

### Linear and Polynomial Regression

- supervised learning
	- supervised: we know solution y to training data
- linear regression
	- solution is linear combination of features
	- least squares
- Gradient descent
	- choose $\theta$ such that $J(\theta)$ is minimal
	- $\theta_{d}:=\theta_{d}-\alpha \frac{\partial}{\partial \theta_{d}}J(\theta)$
	- $\alpha$ is learning rate, simultaneously for all $d = 1, \dots, D$
- Batch Gradient Descent
	- looks at every training sample on every step
	- optimal, but expensive
- Stochastic Gradient Descent
	- randomly select training samples
	- makes progress for each training sample
- Polynomial Curve Fitting
	- model: polynomial
- Generalization and overfitting
	- train error, test error
	- L1/ L2 regression

## Lecture 2

