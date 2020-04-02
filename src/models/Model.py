import torch.nn as nn
import logging
import numpy as np

"""
  src/models/Model.py
  A base class for defining PyTorch {nn.Model} instances
"""

class Model(nn.Module):
    def __init__(self, config=None):
        super(Model, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.classes = None

    def forward(self, *input):
        raise NotImplementedError
        
    def summary(self):
        " Model Summary"
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_params])
        self.logger.info('Trainable Parameters: {}'.format(params))
        self.logger.info(self)

    def __str__(self):
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_params])
        return super(Model, self).__str__() + '\nTrainable params: {}'.format(params)

