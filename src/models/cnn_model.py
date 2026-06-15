import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ECGCNN, self).__init__()
        
        # Input shape: (Batch, 1, 360)
        
        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Conv Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Flattened size: 128 * (360 / 2 / 2 / 2) = 128 * 45 = 5760
        self.fc1 = nn.Linear(128 * 45, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Ensure input is (Batch, 1, 360)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":

    batch_size = 16
    seq_len = 360
    model = ECGCNN(num_classes=2)
    dummy_input = torch.randn(batch_size, seq_len)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
