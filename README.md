# Scene Recognition using Deep Learning
This project defines two convolutional neural networks (CNNs) to classify an input image across 15 categories. The two 
CNN's used are SimpleNet (a custom neural network defined from scratch in this project) and a pre-trained ResNet with 
an extra (linear) classification layer to fit this project's needs.

## SimpleNet
- Source code can be found in src/simple_net.py
- Features two convolutional layers, two pooling layers, and three linear layers
- Very simple CNN that is fast and easy to train

## SimpleNet Modified
- Source code found in src/simple_net_final.py
- Enhanced version of SimpleNet which includes:
  - More layers (convolutional and pooling)
  - Data augmentation to artificially increase number of training examples
  - Data normalization
  - Network regularization (drop out regularization)

## ResNet
- Source Code can be found in src/my_resnet.py
- Takes a pre-trained ResNet CNN and replaces the last layer with a new (untrained) linear layer
- The new linear layer is then trained on our training data for classification
- A multi-label classification ResNet is also defined in this project (multilabel_resnet.py) using the same 15 categories

## Analysis
- Source code includes confusion matrices and plots in order to visualize each CNN's training and testing performance 