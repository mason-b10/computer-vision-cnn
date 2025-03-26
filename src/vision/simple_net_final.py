import torch
import torch.nn as nn


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super(SimpleNetFinal, self).__init__()

        firstNorm = nn.BatchNorm2d(num_features=10)
        firstNorm.weight = nn.parameter.Parameter(torch.ones(10))
        firstNorm.bias = nn.parameter.Parameter(torch.zeros(1))
        secondNorm = nn.BatchNorm2d(num_features=20)
        secondNorm.weight = nn.parameter.Parameter(torch.ones(20))
        secondNorm.bias = nn.parameter.Parameter(torch.zeros(1))

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.BatchNorm2d(num_features=10),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.BatchNorm2d(num_features=20),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(20, 30, kernel_size=5, padding=2),
            nn.MaxPool2d(3, padding=1, stride=1),
            nn.ReLU()
        )


        self.fc_layers = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(750, 100),
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

        return model_output
