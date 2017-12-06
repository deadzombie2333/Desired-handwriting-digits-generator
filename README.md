# Desired-handwriting-digits-generator

Basic idea is use GAN (generative adversarial network) to generate hand writing digit based on requirement. Neural network is trained use MNIST data and Tensorflow.

Generator is design to generate hand writing digits based on user requirement; Discriminator discriminate between real hand writing and "hand writing" generated use generator; Determinator try to decide which digits is displayed.

Generator is use three hidden layer DNN with layer size [100,500,784]. Input of generator is onehot vector（size 1 * 10） with entry "1" at desired digits location and rest of the vector entries are zeros ([0,1,0,0,0,0,0,0,0,0] for desired number 2). This input than transfer to a ( 1 * 100 ) vector where each entry in input vector correspond to 10 entries in the transfered vector with some random variation (multiply input with 10*100 random uniform vector). Idea behind is to get enough randomness in DNN entry for each distinctive digits input to get many random generator outputs. Output of generator is (28 * 28) matrix.

Discriminator and determinator use CNN.

DNN part of generator, discriminator and determinator is regulated use dropout and l1 regulation. Parameters are initialized use xavier initialization with adamoptimizer for minimization.

100 instances per batch, 1000 batches per episode, overall 10 episodes. Generate images after each episode with input as digits 0~9 random shuffles. Image title is the desired digit to display.

![ScreenShot](Desired-handwriting-digits-generator/Figure_1.png)
