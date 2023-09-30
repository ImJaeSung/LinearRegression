import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, layer_size):
        super(RegressionModel, self).__init__()
        self.layer_size = layer_size # list type
        hidden_layers = [
            nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU(inplace = True),
                nn.Dropout(p = 0.2)
                ) 
            for input_size, output_size in zip(self.layer_size, self.layer_size[1:-1])]
        self.hidden_layer = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(self.layer_size[-2], self.layer_size[-1]),
            nn.ReLU(inplace = True)
            )

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)

        return x
