.. -*- mode: rst -*-

neural_network
================

Repository for training neural networks with Keras. 

.. image:: dilated_conv-deconv.png

What is this repository?
------------------------

This repository contains a Python project that can be used to train neural networks with Keras. It also provides several examples with different neural network architectures (such as dilated CNN [1], downsampling/upsampling network [1], hourglass network and deeply recursive network). Experiments reported in the paper [1] can be replicated with the code provided here.

How to use it?
--------------

To use this repository for training a network you must prepare three modules: (1) a data loader, (2) a Keras model and (3) a run file. Below, we explain these modules in detail.

* Data loader - is responsible for preparing data for training and testing the networks. 'data_loader' folder contains two examples of the data loaders 'overlap_data_loader' and 'wsj_data_loader'. 'overlap_data_loader' is developed for the speech overlap detection task and 'wsj_data_loader' is designed for acoustic modeling of an HMM/DNN-based speech recognition on the Wall Street Journal (WSJ) corpus. To write a new data loader you must create a Data class with the following interface:

.. code-block:: python

      from keras.utils import Sequence


      class Data:
          class Subset(Sequence):
              def __init__(self):
                  """
                  constructs a subset that is supposed to manage train, test or development data.
                  It usually defines some fields like self.ids, self.X, self.y and self.shuffle.
                  """
                  pass

              def set_params(self, prm):
                  """
                  Each subset may require some parameters such as batch_size, utterance_input_dim or utterance_output_dim.
                  This function takes these parameters from the prm argument and defines local fields for them.
                  """
                  pass

              def load(self, file_path, shuffle):
                  """
                  loads a subset. You may read all data and store it in the RAM or 
                  you may construct a data structure for reading the data batch-by-batch from the hard disk.
                  """
                  pass

              def on_epoch_end(self):
                  """
                  This function will be called after each epoch.
                  You usually need to shuffle the data (if it is a train set) in this step.
                  """
                  pass

              def __len__(self):
                  """Returns the number of batches per epoch"""
                  pass

              def __getitem__(self, index):
                  """Returns one batch of data"""
                  pass

          def parse_arguments(self, parser):
              """defines the command line arguments that this data provider requires."""
              pass

          def set_params(self, prm):
              self.tr_file = prm.tr_file
              self.de_file = prm.de_file
              self.te_file = prm.te_file
              self.tr.set_params(prm)
              self.de.set_params(prm)
              self.te.set_params(prm)

          def __init__(self):
              self.tr = Data.Subset()
              self.de = Data.Subset()
              self.te = Data.Subset()

          def load(self):
              self.tr.load(self.tr_file, shuffle=True)
              self.de.load(self.de_file, shuffle=False)
              self.te.load(self.te_file, shuffle=False)


* Model - defines the architecture of the neural network. I have written many examples in the model folder. 'conv_net.py', 'dilated_conv.py', 'down_up_net.py' and 'recursive_conv_net.py' contain a standard CNN, a dilated convolutional network, an hourglass network, and a deeply recursive network. To prepare a new model you must write a new class that inherits from KerasNet class. The interface of the class must be as follows:

.. code-block:: python

      from model.keras_net import KerasNet


      class Net(KerasNet):
          def __init__(self):
              super(Net, self).__init__()

          def parse_arguments(self, parser):
              """Adds hyper-parameters of the network to the parser"""
              pass

          def set_params(self, prm):
              """Takes hyper-parameters of the network from the prm argument and stores them in some local fields"""
              pass

          def construct(self):
              """Constructs a Keras network"""
              pass
              
* Run file - determines data loader, model type, hyper-parameters of the model and input/output/log directories. I have prepared some examples in the run directory. For example, 'run_complete_wsj_conv.sh' trains a standard convolutional network for WSJ acoustic modeling on the GPU 0 (CUDA_VISIBLE_DEVICE=0) with a different number of layers (10, 8, 6) and different kernel sizes (9, 7, 5). 


References
----------

.. [1] Soheil Khorram, Zakaria Aldeneh, Dimitrios Dimitriadis, Melvin McInnis, and Emily Mower Provost, 
       *"Capturing Long-Term Temporal Dependencies with Convolutional Networks for Continuous Emotion Recognition"*,
       Interspeech 2017. [`PDF <https://arxiv.org/pdf/1708.07050.pdf>`_]

Author
------

- Soheil Khorram, 2019
