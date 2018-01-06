# NeuroAICodeChallenge



## Discussion

- The neural network I built for numpy seems to perform slightly better than the tensorflow neural network.
- For the tangent nonlinear function, I had to make some alterations. First, I had to shift the given input points by 0.001 because at pi/2 the tangent function is infinite and that was causing the gradient to explode. Second, I had to scale down the impact that the tangent function had on the loss function.


## What I learned from simulated with different numbers of nodes


- In order to test the accuracy of the neural network (MSE), I looked at a finer resolution 0.01 instead of 0.1 to avoid training error
- The plots for all of these experiments are in the current folder
- Both neural networks (numpy and tf) behaved the same way
- For the input nodes, as long as there were more than 3 input nodes, the neural worked fine. There was no noticable improvement as I increased the number of input nodes from 5 to 100. In fact, as I increased the number of input nodes, the complexity of the model increased and with the same number of iterations, the algorithm actually performed worse.
- For the hidden layer nodes, it seems that, so long as I increase the number of iterations and lower the learning rate, the more hidden nodes I add, the better the algorithm performed for all of the nonlinear functions. 
- The universal approximation theorem says that any continuous nonlinear function can be approximated using a single hidden layer with a nonlinearity that follows certain conditions. I think this experiment shows conceptually that the theorem is true, but it may require a large number of nodes in the hidden layer.


## How did C contribute?

- The larger the value of C, the higher the mse or cost functions were.
- The value of C regulates the magnitude of the final network weights
- Qualitatively, C = 1 gives you a very smooth, dampened result that is further from the expected values while C = 0 gives you a spikey result with a smaller error
- I found the C = 0-0.1 to be optimal

- You can see this for yourself in the images in this folder
1. C0.png --> C = 0
2. C1.png --> C = 1
3. Csmall.png --> C = 0.01

- Lower values of C also take longer to converge

## How to test the program

1. Use git to clone the repo
2. Navigate to the directory in your file system
3. Check to make sure your python is version 3
4. run 'python aicode.py' for the numpy code
5. run 'python regression_tf.py' for the tensorflow code

(These may take minute to load)






