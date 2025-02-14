import wandb
from src import build_models, datasets
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from tqdm import tqdm   # for progress bars
import os
import math

# training functionality for pre- and post- projection

def train_log(loss, examples_seen, epoch):
    """Log training progress"""
    wandb.log({"epoch": epoch, "loss": loss}, step=examples_seen)
    print(f'Epoch {epoch}, Examples seen: {examples_seen}, Loss: {loss:.4f}')

def train_model(config, model, train_loader, val_loader, optimizer, criterion, device):

        #add option to save model states and loss/accuracy at nice batch checkpoints

        # Option to save model states and loss/accuracy at checkpoints
        save_checkpoints = config.save_checkpoints
        checkpoint_type = config.checkpoint_type  # 'epoch' or 'batch'
        
        # Training parameters
        epochs = config.epochs

        if save_checkpoints:
            # Create save directory if it doesn't exist
            save_dir = config.save_dir if config.save_dir else './checkpoints'
            os.makedirs(save_dir, exist_ok=True)
            if not os.path.isabs(save_dir):
                save_dir = os.path.abspath(save_dir)
            if checkpoint_type == 'epoch':
                save_dir = os.path.join(save_dir, 'epoch_checkpoints')
            elif checkpoint_type == 'batch':
                save_dir = os.path.join(save_dir, 'batch_checkpoints')
            else:
                raise ValueError('Invalid checkpoint type. Must be "epoch" or "batch"')
            os.makedirs(save_dir, exist_ok=True)
        # Training and validation
        train_losses = []
        train_accuracies = []
        val_accuracies = []

    
        global_batch_counter = 0    # Counter for total number of batches processed
        examples_seen = 0           # Counter for total number of examples processed
        # Determine log-scaled batch intervals
        total_batches = len(train_loader) * epochs
        log_intervals = [int(math.pow(2, i)) for i in range(int(math.log2(total_batches)) + 1)]
        wandb.watch(model, criterion, log='all', log_freq=10)

        for epoch in tqdm(range(epochs)):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                examples_seen += labels.size(0)
                global_batch_counter += 1

                if ((global_batch_counter % 10) == 0):
                    train_log(loss, examples_seen, epoch)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                # Save checkpoint if required
                if save_checkpoints and checkpoint_type == 'batch' and global_batch_counter in log_intervals:
                    checkpoint_path = os.path.join(save_dir, f'checkpoint_batch_{global_batch_counter}.pt')
                    torch.save({
                        'epoch': epoch,
                        'batch': global_batch_counter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': running_loss / (global_batch_counter % len(train_loader)),
                        'accuracy': 100. * correct / total
                    }, checkpoint_path)
                    print(f'Saved checkpoint at batch {global_batch_counter}')
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100. * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_accuracy = 100. * correct / total
            val_accuracies.append(val_accuracy)

            if save_checkpoints and checkpoint_type == 'epoch':
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy
                }, checkpoint_path)
                print(f'Saved checkpoint at epoch {epoch+1}')

            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')
            wandb.log({"train_accuracy": train_accuracy, "val_accuracy": val_accuracy, "train_loss": train_loss}, step=examples_seen)
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

    wandb.log({"test_accuracy": test_accuracy})

    return test_accuracy
     

# train: take in config

def train(config, model, train_loader, val_loader, test_loader, criterion, optimizer):

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    #print device
    print(f"Device: {device}")
    # send model to device
    model = model.to(device)

    # train model
    train_losses, train_accuracies, val_accuracies = train_model(config, model, train_loader, val_loader, optimizer, criterion, device)
    
    # test model
    test_accuracy = test_model(model, test_loader, device)

    return train_losses, train_accuracies, val_accuracies, test_accuracy

## build model from model builder
    # create log file
## train for config.epochs (TODO: add to config)
## log model states/ loss/ accuracy at nice batch checkpoints
## save model to disk
