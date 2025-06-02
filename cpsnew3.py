import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image
from model import VNet
from dataset2 import LAHeart, RandomCrop, RandomNoise, RandomRotFlip, ToTensor

# Dice Loss (Only for Supervised)
def dice_loss(score, target):
    target = target.long()
    score = F.softmax(score, dim=1)
    smooth = 1e-5
    target_onehot = torch.zeros_like(score).scatter_(1, target.unsqueeze(1), 1)
    intersect = torch.sum(score * target_onehot)
    y_sum = torch.sum(target_onehot * target_onehot)
    z_sum = torch.sum(score * score)
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return 1 - dice  # loss

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load subset of labeled data based on the percentage
def get_labeled_data_percentage(dataset, label_percentage=1.0):
    total_labeled_data = len(dataset)
    num_labeled_samples = int(total_labeled_data * label_percentage)
    
    # Randomly sample a subset of labeled data
    sampled_labeled_data = torch.utils.data.Subset(dataset, torch.randperm(total_labeled_data)[:num_labeled_samples])
    
    return sampled_labeled_data

#  Function to load subset of unlabeled data based on the percentage
def get_unlabeled_data_percentage(dataset, unlabeled_percentage=1.0):
    total_data = len(dataset)
    num_samples = int(total_data * unlabeled_percentage)
    sampled_data = torch.utils.data.Subset(dataset, torch.randperm(total_data)[:num_samples])
    return sampled_data

# Model
model_a = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device)
model_b = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).to(device)

# DataLoader setup
batch_size = 4
label_percentage =0.08  # Change this to control the amount of labeled data

train_transform = transforms.Compose([
    RandomRotFlip(),
    RandomNoise(),
    RandomCrop((112, 112, 80)),
    ToTensor(),
])

trainset = LAHeart(split='Training Set', label=True, transform=train_transform)
unlabelled_trainset = LAHeart(split='Training Set', label=False, transform=train_transform)
unlabelled_trainset = get_unlabeled_data_percentage(unlabelled_trainset, unlabeled_percentage=0.50)  #  added line

trainset_subset = get_labeled_data_percentage(trainset, label_percentage)

trainloader = torch.utils.data.DataLoader(trainset_subset, batch_size=batch_size, shuffle=True, num_workers=4)
unlabelled_trainloader = torch.utils.data.DataLoader(unlabelled_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

# Optimizer & Scheduler setup
Max_epoch = 800
learn_rate = 0.0005
optimizer_a = optim.AdamW(model_a.parameters(), lr=learn_rate, weight_decay=1e-4)
optimizer_b = optim.AdamW(model_b.parameters(), lr=learn_rate, weight_decay=1e-4)
scheduler_a = optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=Max_epoch)
scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=Max_epoch)

# TensorBoard setup
writer = SummaryWriter()

# Create directories for saving images
os.makedirs("./images", exist_ok=True)
os.makedirs("./predictions", exist_ok=True)
os.makedirs("./labels", exist_ok=True)

# Helper function to normalize images for saving
def normalize_and_convert_to_image(slice_data):
    min_val = np.min(slice_data)
    max_val = np.max(slice_data)

    # Avoid division by zero if max and min are the same
    if max_val == min_val:
        slice_data = np.zeros_like(slice_data)  # If all pixels are the same, set to black
    else:
        # Normalize to [0, 1] range
        slice_data = (slice_data - min_val) / (max_val - min_val)

    # Scale to [0, 255] and convert to uint8
    slice_data = np.uint8(slice_data * 255)
    return Image.fromarray(slice_data).convert('L')

# Training Loop
for epoch in range(Max_epoch):
    print(f'\nEpoch {epoch+1}/{Max_epoch}')
    print('-' * 30)
    model_a.train()
    model_b.train()

    # ===== Supervised (Labelled) Training =====
    total_sup_loss = 0.0
    total_dice = 0.0

    for batch_idx, sample in tqdm(enumerate(trainloader), total=len(trainloader)):
        optimizer_a.zero_grad()
        optimizer_b.zero_grad()
        images = sample["image"].to(device)
        labels = sample["label"].to(device)

        outputs_a = model_a(images)
        outputs_b = model_b(images)

        seg_loss_a = F.cross_entropy(outputs_a, labels)
        seg_loss_b = F.cross_entropy(outputs_b, labels)
        dice_loss_a = dice_loss(outputs_a, labels)

        sup_loss = seg_loss_a + seg_loss_b + dice_loss_a
        sup_loss.backward()
        optimizer_a.step()
        optimizer_b.step()

        total_sup_loss += sup_loss.item()
        total_dice += (1 - dice_loss_a.item())

    avg_sup_loss = total_sup_loss / len(trainloader)
    avg_dice = total_dice / len(trainloader)

    writer.add_scalar("Supervised Loss", avg_sup_loss, epoch)
    writer.add_scalar("Dice Accuracy/Labelled", avg_dice, epoch)

    # ===== Semi-Supervised (Unlabelled) Training with CPS =====
    if epoch >= 100:
        print(f"--- Starting CPS for Unlabelled Data at Epoch {epoch+1} ---")
        total_unsup_loss = 0.0
        total_unsup_dice = 0.0

        for batch_idx, sample in tqdm(enumerate(unlabelled_trainloader), total=len(unlabelled_trainloader)):
            optimizer_a.zero_grad()
            optimizer_b.zero_grad()
            images = sample["image"].to(device)
            labels = sample["label"].to(device)  # pseudo labels or placeholders

            outputs_a = model_a(images)
            outputs_b = model_b(images)

            _, hardlabel_a = torch.max(outputs_a, dim=1)
            _, hardlabel_b = torch.max(outputs_b, dim=1)

            unsup_cps_loss = 0.01 * (F.cross_entropy(outputs_a, hardlabel_b) + F.cross_entropy(outputs_b, hardlabel_a))
            unsup_cps_loss.backward()
            optimizer_a.step()
            optimizer_b.step()

            # Dice on unlabelled data
            dice_unsup = dice_loss(outputs_a, labels)
            total_unsup_dice += (1 - dice_unsup.item())
            total_unsup_loss += unsup_cps_loss.item()

        avg_unsup_loss = total_unsup_loss / len(unlabelled_trainloader)
        avg_unsup_dice = total_unsup_dice / len(unlabelled_trainloader)

        writer.add_scalar("Unsupervised CPS Loss", avg_unsup_loss, epoch)
        writer.add_scalar("Dice Accuracy/Unlabelled", avg_unsup_dice, epoch)

        # ===== Print Both Labelled & Unlabelled Dice =====
        print(f"Epoch {epoch+1}, Labelled Dice: {avg_dice:.4f}, Unlabelled Dice: {avg_unsup_dice:.4f}, Loss: {avg_unsup_loss:.4f}")

        # ===== Save Images (Original, Predicted, Ground Truth) Every 20 Epochs =====
        if (epoch + 1) % 20 == 0:
            # Convert images and labels to numpy arrays
            image3d = images.detach().cpu().numpy()
            label3d = labels.detach().cpu().numpy()
            pred3d = hardlabel_a.detach().cpu().numpy()

            # Save 2D slices as images (or save a 3D volume if you prefer)
            num_slices = image3d.shape[2]  # Get the total number of slices in Z-dimension
            slice_step = num_slices // 3  # Divide into 3 slices to save

            for i in range(3):  # Save 3 slices (adjust as per your data)
                slice_idx = i * slice_step
                if slice_idx < num_slices:
                    # Ensure slicing is within bounds
                    image_slice = image3d[0][0][:, :, slice_idx]
                    label_slice = label3d[0][:, :, slice_idx]
                    pred_slice = pred3d[0][:, :, slice_idx]

                    # Normalize and convert to images before saving
                    img = normalize_and_convert_to_image(image_slice)
                    img.save(f"./images/{epoch+1}_{i}.png")

                    label_img = normalize_and_convert_to_image(label_slice)
                    label_img.save(f"./labels/{epoch+1}_{i}.png")

                    pred_img = normalize_and_convert_to_image(pred_slice)
                    pred_img.save(f"./predictions/{epoch+1}_{i}.png")

    else:
        # ===== Print Only Labelled Dice Before Epoch 100 =====
        print(f"Epoch {epoch+1}, Labelled Dice: {avg_dice:.4f}, Loss: {avg_sup_loss:.4f}")

    scheduler_a.step()
    scheduler_b.step()

    # Save Model Checkpoints
    if (epoch + 1) % 20 == 0:
        print("Saving model checkpoint...")
        torch.save(model_a.state_dict(), f"model_a_epoch_{epoch+1}.pth")
        torch.save(model_b.state_dict(), f"model_b_epoch_{epoch+1}.pth")

writer.flush()
writer.close()
