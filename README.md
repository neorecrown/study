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

### 11.1.1 Glorot and He initialization ()
Initialization|Activation functions|Normal
:-:|:-:|:-:
Glorot|None,tanh,logistic,softmanx|1\/fanveg
He|ReLU and variants|2\/fanin
LeCun|SELU|1\/fanin

### 11.1.2 Nonsaturating Activation Functions (partially responsible for Gradients problems)
Mother Nature roughly use sigmoid activation functions,while the ReLU performs better because it does note saturate for positive values

**Problem of ReLUs**: dying neurons since the weighted sum of its inputs are negative for all instances

**Evolution of ReLUs**:
1. leaky ReLU (Bing Xu et al., “Empirical Evaluation of Rectified Activations in Convolutional
Network,” arXiv preprint arXiv:1505.00853 (2015).): Leaky $ReLU_\alpha$(z)=max($\alpha$z,z)
2. randomized leaky ReLU(RReLU):$\alpha$ is picked randomly in a given range
3. parametric leaky ReLU (PReLU): $\alpha$ is authorized to be learned during trainning. As reported, the PReLU probably outperform ReLU on large image datasets, but it runs the risk of overfitting on smaller datasets.
4. exponential linear unit(ELU):in the literature (*Djork-Arné Clevert et al., “Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs),” Proceedings of the International Conference on Learning Representations (2016).*), ELU outperformed all the ReLU variants.
$$$ELU_\alpha$(z)=$$. it is slower to compute (exponential function) but has faster convergence rate. Overall slower than a ReLU network
5. Scaled ELU (SELU)(*Günter Klambauer et al., “Self-Normalizing Neural Networks,” Proceedings of the 31st International Conference on Neural Information Processing Systems (2017): 972–981.*): With this activation function, the network will self-normalize, which solves the vanishing/exploding gradients. **Conditions: Standarized inputs, LeCun normal intialization,sequential model, currently suitable for dense layers**

### 11.1.3 Bach Normalization
Functions: Used to zero-centers and normalizes each input, then scales and shifts the result using two new parameter vector per layer
However: makes slower predictions, one solution is to update the weights and bias of the previous layer (TFLite's optimizer).
Position: before or after the activation dependes on task.

### 11.1.4 Gradient Clipping
Denfintion: Clip the gradients during backpropagation.<enter>
<br/>
Basic information: is often used in RNNs, because Batch Norm. is tricky to use in RNNs.<enter>
<br/>
Implement in Keras:  
<br/>
optimizer = keras.optimizers.SGD(clipvalue=1.0), 
<br/>
model.compile(loss='mse', optimizer=optimizer)<enter>
<br/>
Principal：<font color='0000FF'>to be disscussed<\font>
  
 ## 11.2 Reusing Pretrained Layers
 <br/>
 For similar DNN, Trainsfer learning can be very useful.
 <br/>
 we can reuse previous hidden layers and freezing deeper layers to avoid wrecking the fine-tuned weights
 <br/>
 
 ### 11.2.1 Transfer Learning with Keras
 Practice with keras on fashion mnist
 
 
 
 
