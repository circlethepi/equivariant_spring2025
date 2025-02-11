from src import build_models, datasets
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from tqdm import tqdm   # for progress bars

# training functionality for pre- and post- projection

def train_model(args, model, train_loader, val_loader, optimizer, criterion, device):
        # Training parameters
        epochs = args.epochs
        # Training and validation
        train_losses = []
        train_accuracies = []
        val_accuracies = []
    
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100. * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_accuracy = 100. * correct / total
            val_accuracies.append(val_accuracy)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

        return train_losses, train_accuracies, val_accuracies

def test_model(model, test_loader, device):
     # Test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = 100. * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    return test_accuracy
     

# train: take in args

def train(args, model, train_loader, val_loader, test_loader):

    # get optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'Adam' else optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # get loss function
    criterion = nn.CrossEntropyLoss() if args.criterion == 'CrossEntropyLoss' else nn.MSELoss()
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    #print device
    print(f"Device: {device}")
    # send model to device
    model = model.to(device)

    # train model
    train_losses, train_accuracies, val_accuracies = train_model(args, model, train_loader, val_loader, optimizer, criterion, device)
    
    # test model
    test_accuracy = test_model(model, test_loader, device)

    return train_losses, train_accuracies, val_accuracies, test_accuracy

## build model from model builder
    # create log file
## train for args.epochs (TODO: add to args)
## log model states/ loss/ accuracy at nice batch checkpoints
## save model to disk
