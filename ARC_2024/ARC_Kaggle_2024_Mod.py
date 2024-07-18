import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Helper function to resize grids
def resize_grid(grid, size=30):
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    grid_tensor = grid_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    grid_tensor = F.interpolate(grid_tensor, size=(size, size), mode='nearest')
    return grid_tensor.squeeze().numpy().astype(int)

# Load the datasets
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

train_challenges = load_json('data/arc-prize-2024/arc-agi_evaluation_challenges.json')
train_solutions = load_json('data/arc-prize-2024/arc-agi_training_solutions.json')
eval_challenges = load_json('data/arc-prize-2024/arc-agi_evaluation_challenges.json')
eval_solutions = load_json('data/arc-prize-2024/arc-agi_evaluation_solutions.json')
test_challenges = load_json('data/arc-prize-2024/arc-agi_test_challenges.json')

# Preprocess the data
def preprocess_data(challenges, solutions=None, size=30):
    data = []
    for task_id, task in challenges.items():
        for pair in task['train']:
            input_grid = resize_grid(pair['input'], size)
            output_grid = resize_grid(pair['output'], size)
            data.append((input_grid, output_grid))
        if solutions:
            for idx, pair in enumerate(task['test']):
                input_grid = resize_grid(pair['input'], size)
                output_grid = resize_grid(solutions[task_id][idx], size)
                data.append((input_grid, output_grid))
    return data

def preprocess_test_data(challenges, size=30):
    data = []
    for task_id, task in challenges.items():
        for pair in task['test']:
            input_grid = resize_grid(pair['input'], size)
            data.append(input_grid)
    return data

train_data = preprocess_data(train_challenges, train_solutions, size=30)
eval_data = preprocess_data(eval_challenges, eval_solutions, size=30)
test_data = preprocess_test_data(test_challenges, size=30)

# Create custom dataset class
class GridDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

class TestGridDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create data loaders
train_loader = DataLoader(GridDataset(train_data), batch_size=32, shuffle=True)
eval_loader = DataLoader(GridDataset(eval_data), batch_size=32, shuffle=False)
test_loader = DataLoader(TestGridDataset(test_data), batch_size=1, shuffle=False)

# Example CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 30 * 30, 512)
        self.fc2 = nn.Linear(512, 30 * 30)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 30, 30)

# Instantiate and train the model
model = SimpleCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = torch.tensor(inputs).unsqueeze(1).float()
            labels = torch.tensor(labels).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

train_model(model, train_loader, criterion, optimizer, num_epochs=25)

# Predict function
def predict(model, test_loader):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = torch.tensor(inputs).unsqueeze(1).float()
            outputs = model(inputs)
            task_id = list(test_challenges.keys())[i]
            predictions[task_id] = outputs.squeeze().numpy().astype(int).tolist()
    return predictions

# Make predictions
predictions = predict(model, test_loader)

# Prepare submission
with open('submission.json', 'w') as f:
    json.dump(predictions, f)
