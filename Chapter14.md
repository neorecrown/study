# Chapter 14 CNN
## The Architecture of the Visual Cortex
Inspired by the study of structures of the visual cortex

## Convolutional Layers
1. Convolutional Layers connect to pixels in their receptive fields
2. In a CNN, the input images can be 2D
3. CNN, receptive fields: width, height, strides
4. Filters(convolution kernels)
5. Stacking Multiple Feature Maps: Figure 14_6画的不错 <font color="#dd0000">这边需要再对照中文版理解下</font>
6. Tensorflow implementation：keras.layers.Conv2D(filters=, kernel_size=, strids=, padding='', activation='')
8. Memory requirements: <font color="#dd0000">需要再理解下如何计算内存消耗的</font>

## Pooling Layers
1. Performs just like convolutional layers, but without weights, only apllied by an aggregation function such as max or mean.
2. max pooling layers: can provide some translation invariance, even a small amount of rotational invariance and a slight scale invariance.
3. Tensorflow implementation: max_pool/ave_pool = keras.layers.MaxPool2D/AvgPool2D(pool_size=2), strides(default)=kernel size. Generally, max pool performs better than Avg pool, since it enhance the strongest signals. Depthwise max pool: tf.nn.max_pool()

## CNN Architectures
1. Few convonlution layers (ReLUs) --> Pooling layer --> Few convolution layers (ReLUs) --> Pooling layer --> Fully connected (ReLUs)
2. 

