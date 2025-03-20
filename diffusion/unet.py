import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(AudioUNet, self).__init__()

        # Encoder blocks
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder blocks
        self.up4 = self.deconv_block(1024, 512)
        self.up3 = self.deconv_block(512, 256)
        self.up2 = self.deconv_block(256, 128)
        self.up1 = self.deconv_block(128, 64)
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1)
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # Decoder path
        dec4 = self.up4(bottleneck)
        dec4 = self.crop_or_pad(dec4, enc4)  # Crop or pad to match enc4's dimensions
        dec3 = self.up3(dec4)
        dec3 = self.crop_or_pad(dec3, enc3)  # Crop or pad to match enc3's dimensions
        dec2 = self.up2(dec3)
        dec2 = self.crop_or_pad(dec2, enc2)  # Crop or pad to match enc2's dimensions
        dec1 = self.up1(dec2)
        dec1 = self.crop_or_pad(dec1, enc1)  # Crop or pad to match enc1's dimensions
        
        # Final output
        return self.final_conv(dec1)

    def crop_or_pad(self, tensor, target_tensor):
        # This function will ensure the tensor is the same size as the target tensor
        target_size = target_tensor.size()[2:]  # H, W dimensions
        tensor_size = tensor.size()[2:]  # H, W dimensions
        
        # If the sizes are different, we need to crop or pad the tensor
        if tensor_size != target_size:
            # Padding if tensor is smaller
            if tensor_size[0] < target_size[0]:
                padding = (0, target_size[1] - tensor_size[1], 0, target_size[0] - tensor_size[0])
                tensor = F.pad(tensor, padding, mode='constant', value=0)
            # Cropping if tensor is larger
            elif tensor_size[0] > target_size[0]:
                tensor = tensor[:, :, :target_size[0], :target_size[1]]
        return tensor
