import torch
from torch import nn


'''
    .. Combines two models into a single model with two outputs.
'''
class CombinedModel(nn.Module):
    def __init__(self, input_model_crack, input_model_inactive):
        super(CombinedModel, self).__init__()
        self.model_crack = input_model_crack
        self.model_inactive = input_model_inactive

    def forward(self, x):
        output_crack = self.model_crack(x)
        output_inactive = self.model_inactive(x)
        output = torch.cat((output_crack, output_inactive), 1)
        return output