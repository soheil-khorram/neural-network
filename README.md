.. -*- mode: rst -*-

neural_network
================

Repository for training neural networks with Keras. 

.. image:: dilated_conv-deconv.png

What is this repository?
------------------------

This repository contains a Python project that can be used to train neural networks with Keras. It also provides a number of examples with different neural network architectures (such as dilated CNN [1], downsampling/upsampling network [1], hourglass network and deeply recursive network). Experiments reported in the paper [1] can be replicated with the code provided here.

How to use it?
--------------

To use this repository for training a network you must prepare three modules: (1) a data loader, (2) a keras model and (3) a run file. Below, we explain these modules in detail.

* Data loader -- is responsible for preparing data for training and testing the networks. 'data_loader' folder contains two examples of the data loaders 'overlap_data_loader' and 'wsj_data_loader'. 'overlap_data_loader' is developed for the speech overlap detection task and 'wsj_data_loader' is designed for acoustic modeling of a HMM/DNN-based speech recognition on the Wall Street Journal (WSJ) corpus. To write a new data loader you must create a Data class with the following interface:

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
                  This function takes these paramters from the prm argument and defines local fields for them.
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



References
----------

.. [1] Soheil Khorram, Zakaria Aldeneh, Dimitrios Dimitriadis, Melvin McInnis, and Emily Mower Provost, 
       *"Capturing Long-Term Temporal Dependencies with Convolutional Networks for Continuous Emotion Recognition"*,
       Interspeech 2017. [`PDF <https://arxiv.org/pdf/1708.07050.pdf>`_]

Author
------

- Soheil Khorram, 2019

