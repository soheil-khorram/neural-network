.. -*- mode: rst -*-

neural_network
================

Repository for training neural networks with Keras. 
It contains some examples that train standard dilated, downsampling/upsampling and hourglass networks for emotion recognition tasks. Please feel free to contact me if you have any question.

.. image:: dilated_conv-deconv.png

What is this repository?
------------------------

This repository contains a Python project that can be used to train neural networks with Keras. It also provides a number of examples with different neural network architectures (such as dilated CNN, hourglass network and deeply recursive network). 

How to use it?
--------------

To use this repository for training a network you must prepare three modules: (1) a data provider, (2) a keras model and (3) a run file. Below, we explain these modules in detail.



References
----------

.. [1] Soheil Khorram, Zakaria Aldeneh, Dimitrios Dimitriadis, Melvin McInnis, and Emily Mower Provost, 
       *"Capturing Long-Term Temporal Dependencies with Convolutional Networks for Continuous Emotion Recognition"*,
       Interspeech 2017. [`PDF <https://arxiv.org/pdf/1708.07050.pdf>`_]

Author
------

- Soheil Khorram, 2019

