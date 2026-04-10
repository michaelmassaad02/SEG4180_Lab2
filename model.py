import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    A building block used in UNet consisting of:
    Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU
    Optionally includes dropout for regularization.
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            # First convolution layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        
            # Second convolution layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        # Optional dropout to reduce overfitting
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        # Combine all layers into a sequential block
        self.conv = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    UNet architecture for image segmentation.
    Consists of:
    - Encoder (downsampling)
    - Bottleneck
    - Decoder (upsampling with skip connections)
    """
    def __init__(self, dropout=0.4):
        super().__init__()
 
        # Encoder, extract features while reducing spatial size
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
 
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
 
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
 
        # Bottleneck, deepest part of the network capturing high-level feature
        self.bottleneck = DoubleConv(256, 512, dropout=dropout)
 
        # Decoder, restore spatial resolution and combine features from encoder
        # Upsample + concatenate with encoder feature map (skip connection)        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
 
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
 
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        # Final output layer (1 channel for binary segmentation mask)
        self.final = nn.Conv2d(64, 1, kernel_size=1)
 
    def forward(self, x):
        """
        Forward pass through the UNet.
        Includes skip connections between encoder and decoder.
        """
        # Encoder path
        d1 = self.down1(x)
        p1 = self.pool1(d1)
 
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
 
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
 
        # Bottleneck
        b = self.bottleneck(p3)
 
        # Decoder path with skip connections
        # Upsample and concatenate with corresponding encoder features        
        u1 = self.up1(b)
        u1 = torch.cat([u1, d3], dim=1) # Skip connection
        u1 = self.conv1(u1)
 
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1) # Skip connection
        u2 = self.conv2(u2)
 
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1) # Skip connection
        u3 = self.conv3(u3)

        # Final segmentation output (logits)
        return self.final(u3)