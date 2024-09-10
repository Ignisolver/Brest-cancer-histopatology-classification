import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from MAIN.my_constans import DATA_PATH

# 1. Data Loading and Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224 (ResNet-50 input size)
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image data
])

trainset = torchvision.datasets.ImageFolder(root=DATA_PATH / "train", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.ImageFolder(root=DATA_PATH / "test", transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# 2. Model Definition
resnet50 = torchvision.models.resnet50(pretrained=True)
num_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_features, 1)  # Output a single value for binary classification

# 3. Loss Function and Optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
optimizer = optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)

# 4. Training Loop
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50.to(device)

num_epochs = 20

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = resnet50(inputs)
        loss = criterion(outputs.squeeze(), labels.float())  # Squeeze the output to match label shape

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
torch.save(resnet50.state_dict(), 'resnet50_model_state_dict.pth')

# 5. Evaluation
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = resnet50(images)
        predicted = (outputs > 0.5).squeeze().long()  # Convert probabilities to binary predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
