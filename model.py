import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class UNetConv(nn.Module):
	def __init__(self, n_class=1):
		super().__init__()
		# Encoder
		self.enc1 = nn.Sequential(
			nn.Conv3d(1, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv3d(16, 16, kernel_size=3, padding=1),
			nn.ReLU(),
		)
		self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
			
		self.enc2 = nn.Sequential(
			nn.Conv3d(16, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv3d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
		)
		self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
			
		self.enc3 = nn.Sequential(
			nn.Conv3d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv3d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
		)
		self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
			
		self.enc4 = nn.Sequential(
			nn.Conv3d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv3d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
		)
		self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
			
		self.bottleneck = nn.Sequential(
			nn.Conv3d(128, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv3d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(),
		)
			
		# Decoder
		self.upconv4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
		self.dec4 = nn.Sequential(
			nn.Conv3d(256, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv3d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
		)
			
		self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
		self.dec3 = nn.Sequential(
			nn.Conv3d(128, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv3d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
		)
			
		self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
		self.dec2 = nn.Sequential(
			nn.Conv3d(64, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv3d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
		)
			
		self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
		self.dec1 = nn.Sequential(
			nn.Conv3d(32, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv3d(16, 16, kernel_size=3, padding=1),
			nn.ReLU(),
		)
			
		# Output layer
		self.out_conv = nn.Conv3d(16, n_class, kernel_size=1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# Encoder
		enc1 = self.enc1(x)
		enc2 = self.enc2(self.pool1(enc1))
		enc3 = self.enc3(self.pool2(enc2))
		enc4 = self.enc4(self.pool3(enc3))
		
		bottleneck = self.bottleneck(self.pool4(enc4))
		
		# Decoder
		dec4 = self.dec4(torch.cat([self.upconv4(bottleneck), enc4], dim=1))
		dec3 = self.dec3(torch.cat([self.upconv3(dec4), enc3], dim=1))
		dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
		dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))
		
		# Output layer
		return self.sigmoid(self.out_conv(dec1))


class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.25, weight_dice=0.75):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
        
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss

class WeightedBCELoss(nn.Module):
    def __init__(self, beta):
        super(WeightedBCELoss, self).__init__()
        self.beta = beta

    def forward(self, outputs, targets):
        N = targets.size(0)
        y = targets.view(N, -1)
        p = outputs.view(N, -1)

        weight = (1 - self.beta) * (1 - y) + self.beta * y
        bce_loss = -y * torch.log(p) - (1 - y) * torch.log(1 - p)
        loss = torch.mean(weight * bce_loss)

        return loss
