import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import Seq2Emb


class OverfitDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


num_samples = 64
seq_length = 32
vocab_size = 2048
dim = 32
num_layers = 2
dropout = 0
head_dim = 4
hidden_dim = 64

sample_tensors = []
for _ in range(num_samples):
    seq = torch.randint(1, vocab_size, (seq_length,))
    sample_tensors.append(seq)

dataset = OverfitDataset(sample_tensors)
print(f"Dataset size: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=len(dataset))

model = Seq2Emb(vocab_size, dim, num_layers, head_dim, hidden_dim, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.005)
lambda_ = 1e-4

num_epochs = 50_000

# Training
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs.reshape(-1, vocab_size), batch.reshape(-1))
        l2_loss = sum(p.pow(2.0).sum() for p in model.parameters())
        loss2 = loss + lambda_ * l2_loss
        loss2.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Finished Training.")

# Evaluation
with torch.no_grad():
    for batch in dataloader:
        outputs = model(batch)
        _, predicted = torch.max(outputs, dim=-1)
        accuracy = (predicted == batch).float().mean()
        print(f"Reconstruction accuracy: {accuracy.item():.4f}")
        print("Original: ", batch[0])
        print("Predicted:", predicted[0])
        break
