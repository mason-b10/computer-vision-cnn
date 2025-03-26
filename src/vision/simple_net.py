from pyexpat import model
from matplotlib.cbook import flatten
import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        """
        super(SimpleNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(500, 100),
            nn.Linear(100, 15)
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """

        # x = torch.unsqueeze(x, 1)
        conv = self.conv_layers(x)
        model_output = self.fc_layers(conv)

        # ------ TESTING -------
        # print("---- starting foward pass ----")
        # firstConv = self.first_convolve(x)
        # print("made it past first convolve: ")
        # firstConv = self.relu(self.maxpool(firstConv))
        # print("relu and maxpool first conv: ", firstConv.shape)
        # secConv = self.second_convolve(firstConv)
        # print("made it through second convolve: ", secConv.shape)
        # secConv = self.relu(self.maxpool(secConv))
        # print("after relu and maxpool: ", secConv.shape)
        # final_conv = self.flatten(secConv)
        # print("made it through ALL conv layers: ", final_conv.shape)
        

        return model_output
