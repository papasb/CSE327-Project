import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import csv

# ================== CONFIG ==================
DATA_DIR = "/home/bcl/Documents/mj_weather"  # train/val 구조
OUTPUT_DIR = "/home/mjkim2/Documents/Spike_Driven/CNN/output"
NUM_CLASSES = 6
IMG_SIZE = 128
BATCH_SIZE = 32
# LR = 0.01
LR = 0.001
EPOCHS = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== DATA ==================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

train_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"Train set size: {len(train_set)}, Validation set size: {len(val_set)}")
print(f"Number of classes: {NUM_CLASSES}")

# ================== MODEL ==================
model = models.resnet18(pretrained=False, num_classes=NUM_CLASSES)
model = model.to(DEVICE)
print(f"Model: {model}")

# ================== OPTIMIZER & LOSS ==================
# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
print(f"Optimizer: {optimizer}")
print(f"Loss Function: {criterion}")

# ================== LOGGING ==================
log_file = open(os.path.join(OUTPUT_DIR, "log.txt"), "w")
csv_file = open(os.path.join(OUTPUT_DIR, "metrics.csv"), "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Epoch", "Loss", "Top1 Acc", "Top5 Acc"] + [f"Class_{i}_Acc" for i in range(NUM_CLASSES)])

# ================== HELPER ==================
def per_class_accuracy(y_true, y_pred, num_classes):
    correct = [0] * num_classes
    total = [0] * num_classes
    for t, p in zip(y_true, y_pred):
        total[t] += 1
        if t == p:
            correct[t] += 1
    return [correct[i]/total[i] if total[i] > 0 else 0.0 for i in range(num_classes)]

def log_and_print(message):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()


# ================== TRAIN & VALIDATE ==================
for epoch in range(1, EPOCHS+1):
    log_and_print(f"\nEpoch {epoch}/{EPOCHS}")
    # -------- TRAIN --------
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        _, pred_top5 = outputs.topk(5, 1, True, True)
        correct_top1 += (pred_top5[:, 0] == labels).sum().item()
        correct_top5 += sum([labels[i] in pred_top5[i] for i in range(labels.size(0))])

        if batch_idx % 500 == 0 or batch_idx == len(train_loader):
            log_and_print(f"  [Batch {batch_idx}/{len(train_loader)}]")


    avg_loss = running_loss / total
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    # -------- VALIDATE --------
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    class_acc = per_class_accuracy(all_labels, all_preds, NUM_CLASSES)

    # -------- LOGGING --------
    log_file.write(f"Epoch {epoch}: Loss={avg_loss:.4f}, Top1={top1_acc:.4f}, Top5={top5_acc:.4f}\n")
    for i, acc in enumerate(class_acc):
        log_file.write(f"Class {i} Acc: {acc:.4f}\n")
    log_file.flush()

    csv_writer.writerow([epoch, avg_loss, top1_acc, top5_acc] + class_acc)
    csv_file.flush()

log_file.close()
csv_file.close()
print("Training complete! Logs saved.")
