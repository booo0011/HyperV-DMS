import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
from core.hgnn_model import ResumeVoyagerNet

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

class SyntheticMultimodalDataset(Dataset):
    def __init__(self, size=200, v_dim=956, a_dim=40, num_classes=5):
        self.v_dim = v_dim
        self.a_dim = a_dim
        self.num_classes = num_classes
        self.size = size
        self.data = [(
            torch.randn(v_dim, dtype=torch.float32),
            torch.randn(a_dim, dtype=torch.float32),
            torch.randint(0, num_classes, (1,), dtype=torch.long).item()
        ) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        v_feat, a_feat, label = self.data[idx]
        return v_feat, a_feat, label

class BaselineConcatNet(nn.Module):
    def __init__(self, v_dim=956, a_dim=40, hidden_dim=128):
        super().__init__()
        self.v_proj = nn.Linear(v_dim, hidden_dim)
        self.a_proj = nn.Linear(a_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)
        )

    def forward(self, v_feat, a_feat):
        vh = torch.relu(self.v_proj(v_feat))
        ah = torch.relu(self.a_proj(a_feat))
        combined = torch.cat([vh, ah], dim=1)
        return self.classifier(combined)


def train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for v_batch, a_batch, labels in train_loader:
            v_batch = v_batch.to(device)
            a_batch = a_batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(v_batch, a_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch [{epoch+1}/{epochs}] loss: {avg_loss:.4f}")
    return model


def evaluate_model(model, data_loader, device):
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for v_batch, a_batch, labels in data_loader:
            v_batch = v_batch.to(device)
            a_batch = a_batch.to(device)
            outputs = model(v_batch, a_batch)
            _, predicted = torch.max(outputs, dim=1)
            preds.extend(predicted.cpu().tolist())
            actuals.extend(labels.tolist())
    acc = sum(p == t for p, t in zip(preds, actuals)) / max(len(actuals), 1)
    f1 = f1_score(actuals, preds, average='macro', zero_division=0)
    return acc, f1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = SyntheticMultimodalDataset(size=200)
    test_dataset = SyntheticMultimodalDataset(size=80)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    baseline = BaselineConcatNet().to(device)
    resume_model = ResumeVoyagerNet().to(device)

    baseline_optimizer = optim.Adam(baseline.parameters(), lr=1e-3, weight_decay=1e-4)
    baseline_scheduler = StepLR(baseline_optimizer, step_size=2, gamma=0.5)
    resume_optimizer = optim.Adam(resume_model.parameters(), lr=1e-3)
    resume_scheduler = StepLR(resume_optimizer, step_size=2, gamma=0.5)

    criterion = nn.CrossEntropyLoss()

    print("\nTraining BaselineConcatNet...")
    train_model(baseline, train_loader, criterion, baseline_optimizer, baseline_scheduler, device, epochs=5)

    base_acc, base_f1 = evaluate_model(baseline, test_loader, device)
    print(f"Baseline accuracy: {base_acc:.4f}, macro F1: {base_f1:.4f}")

    print("\nTraining ResumeVoyagerNet...")
    train_model(resume_model, train_loader, criterion, resume_optimizer, resume_scheduler, device, epochs=5)

    resume_acc, resume_f1 = evaluate_model(resume_model, test_loader, device)
    print(f"ResumeVoyagerNet accuracy: {resume_acc:.4f}, macro F1: {resume_f1:.4f}")

    save_path = os.path.join(MODEL_DIR, 'synthetic_resume_voyagernet.pth')
    torch.save(resume_model.state_dict(), save_path)
    print(f"Saved trained model to {save_path}")


if __name__ == '__main__':
    main()
