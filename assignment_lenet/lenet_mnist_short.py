import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data preprocessing
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Original LeNet-5 model
class LeNet(nn.Module):
    def __init__(self, activation=nn.ReLU, pool=nn.MaxPool2d):
        super(LeNet, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.activation1 = activation()
        self.pool1 = pool(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.activation2 = activation()
        self.pool2 = pool(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.activation3 = activation()
        self.fc2 = nn.Linear(120, 84)
        self.activation4 = activation()
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        
        x = x.view(-1, 16 * 5 * 5)
        
        x = self.fc1(x)
        x = self.activation3(x)
        x = self.fc2(x)
        x = self.activation4(x)
        x = self.fc3(x)
        
        return x

# LeNet without pooling (to test the importance of pooling)
class LeNetNoPooling(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super(LeNetNoPooling, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=2, padding=2)  # Stride 2 to replace pooling
        self.activation1 = activation()
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=2)  # Stride 2 to replace pooling
        self.activation2 = activation()
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.activation3 = activation()
        self.fc2 = nn.Linear(120, 84)
        self.activation4 = activation()
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.activation2(x)
        
        x = x.view(-1, 16 * 5 * 5)
        
        x = self.fc1(x)
        x = self.activation3(x)
        x = self.fc2(x)
        x = self.activation4(x)
        x = self.fc3(x)
        
        return x

# Function to train the model
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {running_loss/(batch_idx+1):.3f}, Acc: {100.*correct/total:.3f}%')
    
    train_time = time.time() - start_time
    accuracy = 100. * correct / total
    
    return running_loss / len(train_loader), accuracy, train_time

# Function to evaluate the model
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return test_loss / len(test_loader), accuracy

def main():
    # Load data
    train_loader, test_loader = load_data()
    criterion = nn.CrossEntropyLoss()
    
    # Define model configurations to test
    models = {
        "LeNet_ReLU_MaxPool": LeNet(activation=nn.ReLU, pool=nn.MaxPool2d),
        "LeNet_Sigmoid_MaxPool": LeNet(activation=nn.Sigmoid, pool=nn.MaxPool2d),
        "LeNet_NoPooling_ReLU": LeNetNoPooling(activation=nn.ReLU)
    }
    
    # Results tracking
    results = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n===== Training {model_name} =====")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        training_times = []
        
        for epoch in range(1, 3):  # Train for only 3 epochs for quick testing
            train_loss, train_acc, train_time = train(model, train_loader, optimizer, criterion, device, epoch)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            training_times.append(train_time)
            
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {train_time:.2f}s')
        
        results[model_name] = {
            'train_loss': train_losses,
            'test_loss': test_losses,
            'train_acc': train_accs,
            'test_acc': test_accs,
            'final_test_acc': test_accs[-1],
            'avg_train_time': sum(training_times) / len(training_times)
        }
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    for model_name, result in results.items():
        plt.plot(range(1, len(result['test_acc'])+1), result['test_acc'], marker='o', label=model_name)
    
    plt.title('Test Accuracy vs Epochs for Different Model Configurations')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('accuracy_comparison_short.png')
    
    # Print summary
    print("\n===== Summary of Results =====")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_test_acc'], reverse=True)
    
    for i, (model_name, result) in enumerate(sorted_results):
        print(f"{i+1}. {model_name}: Test Acc = {result['final_test_acc']:.2f}%, Avg Training Time = {result['avg_train_time']:.2f}s")

if __name__ == "__main__":
    main() 