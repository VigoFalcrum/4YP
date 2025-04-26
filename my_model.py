# my_model.py
import torch
import torch.nn as nn

class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, num_classes):
        super(DeepNN, self).__init__()
        layers = []

        # First layer: from input to hidden_size
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers: hidden_size -> hidden_size
        # Ensure num_hidden_layers includes input and output layer considerations
        # If num_hidden_layers is total layers, loop should be num_hidden_layers - 2
        # If num_hidden_layers means hidden layers *between* first and last, loop is num_hidden_layers
        # Assuming num_hidden_layers = total number of Linear layers:
        hidden_layer_count = max(0, num_hidden_layers - 2) # Adjust logic if definition differs
        for _ in range(hidden_layer_count):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Final layer: from hidden_size to num_classes
        layers.append(nn.Linear(hidden_size, num_classes))
        # Note: No final activation (like Softmax) needed here if using nn.CrossEntropyLoss during *training*
        # or if post-processing handles activation outside the model.

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)