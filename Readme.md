An implementation of the Parallel Restarted SGD algorithm (PR-SGD) described in the paper "Parallel restarted SGD with faster convergence and less communication: demystifying why model averaging works for deep learning" 
https://arxiv.org/abs/1807.06629

PR-SGD is a distributed variant of SGD, where we have $k$ workers, each doing a local stochastic gradient descent for $I_i$ steps (numLocalSteps[i] in the code), then their parameters are averaged and new local stochastic gradient descents are done using the new parameters array. 

The workers can have different loss functions, and PR-SGD is proved to cenverge to a local minima of the average loss function (under conditions on the functions).

- Program.cs is the main file, we specify in it the number of workers, the loss functions and other hyper-parameters  

- LossFunction.cs gives an interface for defining loss functions, each object from a class implementing this interface should be able to return a value and a gradient when given an array vector with the corresponding inputDimension. We also define in the file two types of loss functions: QuadraticLOss and AverageLOss  

- Core.cs: responsible for sending requests to the workers and synchronizizng their results

- Worker.cs: defines a worker having a local parameters array and an objective function, capable of doing local gradient descents