# Anomaly detection using Isolation Forest

## Introduction

This repository includes an implementation of Isolation Forest in Python and some examples on how it is used. While it is still significantly slower than C/C++ or java implementations we are able to get a significant bust by taking advantage of NumPy (i.e. running stuff in C 😜).

## Developement

- [ ] Algorithm
  - [ ]
- [ ] Example
  - [ ]
- [ ] Images


## The algorithm

Anomaly detection is largely needed nowadays. One of the fields were this is most clear is CyberSecurity. Up until the data K-Nearest Neighbors was the usual "way to go" when dealing with anomaly detection when dealing with un-labelled data.

That was working great with continuous variable but when categorical features were thrown in the game either some weird pre-processing was needed or you had to come up with a specific metric for those "categorical variable dimensions".

[Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest) is an algorithm fruit of an intuitive idea that overcame the above complications.

The basic idea is the following:

> Imagine that you are in a soccer field full of children. They are all evenly distributed among the field at the beginning. When the game starts, the children tend to go where the ball is located. But there is always that one weird child that remains in its position without chasing the ball. This kid is an anomaly. 

<p align="center">
  <img src="https://github.com/r0mer0m/anomaly-detection-isolation-forest/blob/master/images/field_1.png">
</p>

> If we **randomly select one of the sides** and **randomly split it** we will have a partition of the children (if we have more dimensions we would make that split in more than one dimensions and see which split separates the data in the most unbalanced way).

<p align="center">
  <img src="https://github.com/r0mer0m/anomaly-detection-isolation-forest/blob/master/images/field_2.png">
</p>

> We will do that again and again until a pre-defined max depth.

<p align="center">
  <img src="https://github.com/r0mer0m/anomaly-detection-isolation-forest/blob/master/images/field_3.png">
</p>

> The weird child has been isolated sooner than the others! That's the basic idea of the algorithm. 

A bit more technical notes:

* Observe that (all) those splits are defining a tree. By repeating that process N times we get N trees i.e. a forest. 

* The more observation is an anomaly the sooner it will be isolated in the different trees. To compute "how soon" the authors use some maths that takes into account the maximum expected length of the trees. As a side note, the splits in each dimension are computed using a uniform / discrete uniform distribution.

This example was provided by [Terrence Part](https://github.com/parrt) in the [MSDS at the University of San Francisco](https://www.usfca.edu/arts-sciences/graduate-programs/data-science).

