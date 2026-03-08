import torch.nn as nn

class MLPV1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPV1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )
    
    def forward(self, x):
        return self.model(x)

class MLPV2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPV2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )
    
    def forward(self, x):
        return self.model(x)
        