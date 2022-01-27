import torch
import torch.nn as nn

class ConvolutionalNeuralNetwork(nn.Module):
        def __init__(self):
            super(ConvolutionalNeuralNetwork, self).__init__()
            self.cnn_net = nn.Sequential(
                nn.Conv2d(1, 8, 3),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3),
                nn.ReLU(),
                nn.Conv2d(16, 8, 3),
                nn.ReLU()
                )
            
            self.classifier = nn.Sequential(
                nn.Linear(8*22*22, 1000),
                nn.ReLU(),
                nn.Linear(1000, 10)
                )
            
        def forward(self, x):
            x = self.cnn_net(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x