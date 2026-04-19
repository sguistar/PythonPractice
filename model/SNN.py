import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

class SpikingNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super(SpikingNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay
        self.membrane_potential = 0

    def forward(self, x):
        self.membrane_potential += x
        spike = (self.membrane_potential >= self.threshold).float()
        return spike

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = SpikingNeuron(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

X = torch.randn(1000, 2)
y = (X[:,0] + X[:,1] > 0).float()

dataset = data.TensorDataset(X, y)
dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True)

model = SNN(input_size=2, hidden_size=10, output_size=1)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
NUM_EPOCHS = 300
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in dataloader:
        outputs = model(X_batch)
        loss = criterion(outputs.view(-1), y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        correct += ((outputs.view(-1) > 0) == y_batch).sum().item()
        total += y_batch.size(0)

    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/total:.4f}, Accuracy: {correct/total:.4f}')

X_test = torch.randn(10, 2)
y_test = (X_test[:,0] + X_test[:,1] > 0).float()

with torch.no_grad():
    outputs = model(X_test)
    test_loss = criterion(outputs.view(-1), y_test)
    test_accuracy = ((outputs.view(-1) > 0) == y_test).float().mean()
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')