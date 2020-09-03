# Genetic-Algorithm

This repository is the implementation of genetic algorithm for Neural Network Weight Optimization in python using scikit-learn and numpy as helper libraries.

## Neural Network Architecture

I have used Iris Dataset for training which is a standard dataset having 4 attributes and 3 output classes **(multi-class classification)**. The total number of training examples is 150.

- Number of Input Neurons = 4
- Number of Hidden Layers = 1
- Number of Hidden Neurons = 4
- Number of Output Neurons = 3

Thus, the total number of genes in a chromosome is obtained as: 4x4 + 3x4 + 4 + 3 = 35

## Graphical Outputs/Statistics

**Plot of Fitness Values --** <br />
![Plot of Fitness Values](/images/Figure_1.png)

**Mean Loss vs Iterations --** <br />
![Mean Loss vs Iterations](/images/Figure_2.png)

**Accuracy vs Iterations --** <br />
![Accuracy vs Iterations](/images/Figure_3.png)
