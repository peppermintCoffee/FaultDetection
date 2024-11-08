import numpy as np
import matplotlib.pyplot as plt
import segyio
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall
from utils import SeismicDataset
from model import *
import os
import random
os.environ["WANDB_SILENT"] = "True"
import wandb
wandb.login()

torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


def calculate_pos_weight(train_loader):
	pos_weight = []

	for _, targets in train_loader:
		total_positive = targets.sum().item()
		total_negative = (targets == 0).sum().item()
		pos_weight.append(total_negative / total_positive)

	return torch.tensor([pos_weight])

def train_model(model, train_loader, val_loader, num_epochs, device, lr):
	wandb.init(project = "fault-detection")

	# pos_weight = calculate_pos_weight(train_loader).to(device)
	# criterion = nn.BCELoss(weight=pos_weight).to(device)
	criterion = BCEDiceLoss().to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.9, threshold=0.0005, min_lr=1e-7, verbose=True)

	iou_metric = BinaryJaccardIndex().to(device)
	f1_metric = BinaryF1Score().to(device)

	best_val_iou = 0.0
	best_model_path = f'best_model.pth'

	for epoch in range(num_epochs):
		model.train()
		train_loss = 0
		iou_metric.reset()
		f1_metric.reset()

		for inputs, targets in train_loader:
			inputs, targets = inputs.to(device), targets.to(device)

			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			iou_metric.update(outputs, targets)
			f1_metric.update(outputs, targets)

		train_loss /= len(train_loader)
		train_iou = iou_metric.compute().item()
		train_f1 = f1_metric.compute().item()

		model.eval()
		val_loss = 0
		iou_metric.reset()
		f1_metric.reset()

		with torch.no_grad():
			for inputs, targets in val_loader:
				inputs, targets = inputs.to(device), targets.to(device)
				outputs = model(inputs)
				loss = criterion(outputs, targets)
				val_loss += loss.item()
				iou_metric.update(outputs, targets)
				f1_metric.update(outputs, targets)

		val_loss /= len(val_loader)
		val_iou = iou_metric.compute().item()
		val_f1 = f1_metric.compute().item()

		scheduler.step(val_loss)

		wandb.log({
			"epoch": epoch,
			"train_loss": train_loss,
			"train_iou": train_iou,
			"train_f1": train_f1,
			"val_loss": val_loss,
			"val_iou": val_iou,
			"val_f1": val_f1
		})

		if val_iou > best_val_iou:
			best_val_iou = val_iou
			torch.save(model.state_dict(), best_model_path)
	
		print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
			  f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

	wandb.finish()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = SeismicDataset(dpath='data/train/seis/', fpath='data/train/fault/', augment_factor=3)
val_dataset = SeismicDataset(dpath='data/validation/seis/', fpath='data/validation/fault/', augment_factor=0)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

model = UNetConv().to(device)
# train_model(model, train_loader, val_loader, num_epochs=150, device=device, lr=1e-4)

# import torchinfo

# print(torchinfo.summary(model, (1, 1, 128, 128, 128), col_names = ("input_size", "output_size", "kernel_size"), verbose = 0))
