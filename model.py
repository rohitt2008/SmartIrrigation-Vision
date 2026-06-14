import torch
import torch.nn as nn

class LightResidualBlock(nn.Module):
    """
    Modified lightweight residual block as shown in Fig. 3(c):
    Main branch: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
    Shortcut branch: Conv1x1 -> BN
    Addition: Main + Shortcut -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(LightResidualBlock, self).__init__()
        # Main branch
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut branch (Conv 1x1 + BN)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)
        return out

class LightResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super(LightResNet50, self).__init__()
        
        # Input Block (I1)
        # Table II: Block-A
        # A1: Conv2D(16, 3x3, stride 2) + BN + ReLU -> (112, 112, 16)
        self.conv_a1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_a1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # A2: Conv2D(16, 3x3, stride 2) + BN + ReLU -> (56, 56, 16)
        self.conv_a2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_a2 = nn.BatchNorm2d(16)
        
        # A3: MaxPool2D(2, 2) -> (28, 28, 16)
        self.pool_a = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block-B: Input (28, 28, 16) -> Output (14, 14, 32)
        self.block_b = LightResidualBlock(in_channels=16, out_channels=32)
        self.pool_b = nn.MaxPool2d(kernel_size=2, stride=2)  # B5: MaxPool2D -> (14, 14, 32)
        
        # Block-C: Input (14, 14, 32) -> Output (7, 7, 64)
        self.block_c = LightResidualBlock(in_channels=32, out_channels=64)
        self.pool_c = nn.MaxPool2d(kernel_size=2, stride=2)  # C5: MaxPool2D -> (7, 7, 64)
        
        # Block-D: Input (7, 7, 64) -> Output (3, 3, 128)
        self.block_d = LightResidualBlock(in_channels=64, out_channels=128)
        self.pool_d = nn.MaxPool2d(kernel_size=2, stride=2)  # D5: MaxPool2D -> (3, 3, 128)
        
        # Block-E: Input (3, 3, 128) -> Output (1, 1, 256)
        self.block_e = LightResidualBlock(in_channels=128, out_channels=256)
        self.pool_e = nn.MaxPool2d(kernel_size=3, stride=3)  # E5: MaxPool2D -> (1, 1, 256)
        
        # Output block
        # F1: Flatten -> 256
        # F2: Dense -> 64 (ReLU)
        # F3: Dense -> 4 (LogSoftmax or Raw Logits depending on loss)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Block-A
        x = self.relu(self.bn_a1(self.conv_a1(x)))
        x = self.relu(self.bn_a2(self.conv_a2(x)))
        x = self.pool_a(x)
        
        # Block-B
        x = self.block_b(x)
        x = self.pool_b(x)
        
        # Block-C
        x = self.block_c(x)
        x = self.pool_c(x)
        
        # Block-D
        x = self.block_d(x)
        x = self.pool_d(x)
        
        # Block-E
        x = self.block_e(x)
        x = self.pool_e(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully Connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # outputs raw logits
        
        return x

if __name__ == "__main__":
    # Test shape check
    model = LightResNet50(num_classes=4)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Model initialized successfully.")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    
    # Print trainable parameters count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params:,} ({params/1e6:.2f}M)")
