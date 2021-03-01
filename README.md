# chapter 11 Training Deep Neural Networks
## Problems you could run into:
1. vanishing gradients, exploding gradients
2. not enough labeled data (costly)
3. slow trainning
4. overfitting (less train data while complex DNN)

## 11.1 The Vanishing/Exploding Gradients Problems (Major reason for abandoning DNN in the early 2000s)
### Denfision:
Vanishing gradients: Gradients get smaller and smaller as the algorithm down to the lower layersm, thus the gradient descent update leaves the lower layers' connection weights virtually unchanged, and trainning never converges to a good solution.
Exploding gradients: Gradients grow bigger and bigger until layers get insanely large weight updates.

### Glorot and He initialization
Initialization|Activation functions|Normal
:-:|:-:|:-:
Glorot|None,tanh,logistic,softmanx|1\/fanveg
He|ReLU and variants|2\/fanin
LeCun|SELU|1\/fanin

### Nonsaturating Activation Functions (partially responsible for Gradients problems)
Mother Nature roughly use sigmoid activation functions,while the ReLU performs better because it does note saturate for positive values
**Problem of ReLUs**: dying neurons since the weighted sum of its inputs are negative for all instances
