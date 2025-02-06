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

### Scaling

- Dennard Scaling
	- same space more capacitors
		- density $\alpha^{2}$
		- Lower voltage, capacity (each$\frac{1}{\alpha}$)
		- Higher speed ($\alpha$)
		- Active power density (1)
		- perf = instructions per cycle times frequency
	- Reality: Post-Dennard
		- wiring + leakage
		- voltage scaling no longer possible
		- power no longer remains constant
		- perf = power times efficiency
		- transition to massively parallel microarchitectures
- vector isas
	- single instruction, n operations
	- parallel: n operations are data parallel
	- expressive: memory operations describe patterns
- our view of a GPU
	- Software view: many core scalar architecture
	- SIMT: single instruction, multiple threads
	- hardware view: multi-core vector architecture
	- SIMD: single instruction, multiple data
	- vector architecture that hides vector units
- Bulk-synchronous parallel
	- Superstep: compute, communicate, synchronize
	- v >> p
	- extremely scalable, bad for unbalanced parallelism
- Simplicity
	- Thread ID determines data element
	- One thread per output element
	- One thread per data element
- Memory: TODO: look at this again

## Lecture 3

### ANN

- DNN = increasing number of hidden layers
- $\mathbf{x}_{l} = f(\mathbf{W}_{l}\cdot\mathbf{x}_{l-1}+\mathbf{b}_{l})$
- non-linear functions
- SoftMax for output
- Gradient behaviour
	- add gate: gradient distributor
	- mul gate: gradient switcher
	- max gate: gradient router
- Convolution Layers
	- spatially local correlation
	- shared weights
	- 3d layers

## Lecture 4

### Autograd

- Computational graph
- topological ordering
- $\frac{\partial \mathscr{L}}{\partial v} = \frac{\partial \mathscr{L}}{\partial u} \frac{\partial u}{\partial v} = \bar{u}\frac{\partial u}{\partial v} = \bar{v}$
- $\mathbf{\bar{x}} = \mathbf{J}^T\mathbf{\bar{y}}$
- $\mathbf{y} = \mathbf{W}\mathbf{x} \implies \mathbf{\bar{x}} = \mathbf{W}^T\mathbf{\bar{y}}$
- $\mathbf{\bar{W}} = \mathbf{\bar{y}} \mathbf{x}^T = \mathbf{\bar{y}} \otimes \mathbf{x}$
- $\mathbf{y} = e^{\mathbf{x}} \implies \mathbf{\bar{x}}=e^\mathbf{x} \circ \mathbf{\bar{y}}$
- Autograd Algorithm
	- Let $v_{1},\dots,v_{N}$ denote all vertices
	- Let Pa($v_{i}$) be the parents of $v_{i}$ , Ch($v_{i}$) be the children of $v_{i}$
	- For $i = 1,\dots, N$ Compute $v_{i}$ as function of Pa($v_{i}$)
	- Set $\bar{v}_{N}=1$ by convention
	- For $i = N-1, \dots, 1$
		$\bar{v}_{i}=\sum_{j \in \text{Ch}(v_{i})}\mathbf{v}_{j} \frac{dv_{j}}{dv_{i}}$

### Gradient Descent

- Variants
	- Batch Gradient Descent
		- slow and memory/ compute intensive
		- guaranteed to be optimal, bad for large datasets
	- Stochastic Gradient Descent
		- overcomes local minima/ saddle points
		- usually small learning rates
	- Mini-batch Stochastic Gradient Descent
		- reduces the variance
		- mini-batch size often a power of two
- Learning Rates
	- Learning rate schedules
	- momentum
	- adagrad (adapts the learning rate to the parameters)
	- adadelta/ rmsprop (same as adagrad but only considers some subset of previous learning rates)
	- adam (adadelta/rmsprop + momentum)

## Lecture 5

### Regularization

- underfitting/ overfitting
- add a penalty term to the error function
- No free lunch theorem, occams razor
- regularization is any modification of a learning method that reduces its generalization error but not the training error
- expectation/ variance
- many regularization methods shrink coefficient estimates towards zero
- prevents the learning of complex models
- penalize the flexibility of a model
- examples: shrinkage methods, early stopping, dropout, weight initialization, batch normalization

### Examples

- early stopping
	- stop when test error starts increasing
	- difficult due to error fluctuations
- data augmentation
	- depends on task
	- transforms
	- only augment the training set
	- way longer training
- reduce number of parameters
	- bottleneck layers
- L2 norm/ weight decay
	- shrink weights by a factor
- L1 norm/ lasso
	- l1 norm instead of l2
- Ensembles
	- average multiple models
	- multiple independent training sets
- Dropout
	- wight probability $\rho$, set output activation to zero
	- compensate at test time by multiplying with $1-\rho$
	- dropout transforms a single network into an ensemble

often pruning/ quantization can make models more general

## Lecture 6

- batch normalization
	- dont use in combination with dropout
	- reduce covariate shift between training and test set
	- mitigates interdependency between layers during training
	- LN, IN, GN
- convolutional architectures
	- pooling
		- max pooling
		- average pooling
	- TODO: naves of variables

## Lecture 7

- unsafe optimization
	- optimizations that affect accuracy
	- quantized neural networks
		- piece-wise constant function
		- TODO: implications
	- pruning
		- remove stuff when its useless
			- remove smallest weights
			- remove weights below threshold
			- multiply by gradient before threshold

## Lecture 8

## Lecture 9

- FINN
	- deploy DNNs to chips
- Column Pruning
	- coarse-grained: blocks of matrix column
	- fine-grained: pruning of single matrix columns

## Lecture 10

- cpus use speculation everywhere
- DNNs don't need speculation, branch prediction, cache agents, OOO, Multi-threading
- performance vs innovation tradeoff