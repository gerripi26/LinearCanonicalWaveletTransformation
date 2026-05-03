import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from LCWT_model import LCWT

"""
github: git add .
git commit -m ""
git push origin main
"""

# hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8

# training
n_epochs = 4
eval_iter = 50

class ECG_Dataset(Dataset):
    def __init__(self, data, labels):
        self.X = data
        self.labels = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.labels[idx]
        mean = x.mean()

        # normalize
        std = x.std() + 1e-8
        x = (x - mean) / std

        return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    correct = 0
    total = 0
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iter, device=device)
        for it, (xb, yb) in enumerate(loader):
            if it == eval_iter:
                break

            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            losses[it] = loss
            predicted_classes = pred.argmax(dim=1)
            correct += (predicted_classes == yb).sum().item()
            total += yb.size(0)

        out[split] = losses.mean()
        #accuracy = correct / total
        #print("Validation Accuracy:", accuracy)
        xb = xb.to(device)
        model(xb, plot=True)

    model.train()
    return out

# read file
train_file = pd.read_csv('/ECG_data/mitbih_train.csv')
test_file = pd.read_csv('/ECG_data/mitbih_test.csv')

# Train and test splits
train_data = torch.tensor(train_file.values, dtype=torch.float32)[:, :-2]  # (samples, T + class label)
train_data_labels = torch.tensor(train_file.values, dtype=torch.long)[:, -1]
val_data = torch.tensor(test_file.values, dtype=torch.float32)[:, :-2]
val_data_labels= torch.tensor(test_file.values, dtype=torch.long)[:, -1]

# Dataset, Dataloader
train_dataset = ECG_Dataset(train_data, train_data_labels)
val_dataset = ECG_Dataset(val_data, val_data_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# create model
model = LCWT()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# training loop
model.train()
for epoch in range(n_epochs):
    for it, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % eval_iter == 0:
            losses = estimate_loss()
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")