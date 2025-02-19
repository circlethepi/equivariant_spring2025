import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random
import wandb

class RotatedDataset(Dataset):
    def __init__(self, original_dataset, angles):
        self.original_dataset = original_dataset
        self.angles = angles
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
        ])

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        angle = random.choice(self.angles)
        rotated_img = self.transform(img).rotate(angle)
        rotated_img = transforms.ToTensor()(rotated_img)
        return rotated_img, label

def rotate_tensor(tensor, angle):
    pil_image = transforms.ToPILImage()(tensor)
    rotated_image = pil_image.rotate(angle)
    return transforms.ToTensor()(rotated_image)

def random_rotate_dataset(dataloader, angles=[0, 90, 180, 270]):
    original_dataset = dataloader.dataset
    rotated_dataset = RotatedDataset(original_dataset, angles)
    return DataLoader(rotated_dataset, batch_size=dataloader.batch_size, shuffle=False, num_workers=4, pin_memory=True)

def average_over90degrees_and_evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = []
            # Rotate inputs by 0, 90, 180, 270 degrees and run through the model
            for angle in [0,90,180,270]:
                rotated_inputs = torch.stack([rotate_tensor(img, angle) for img in inputs])
                rotated_inputs = rotated_inputs.to(device)
                output = model(rotated_inputs)
                outputs.append(output)

            # Average the outputs and divide by 4
            averaged_output = sum(outputs) /4
            _, predicted = averaged_output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_accuracy = 100. * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    wandb.log({"test_accuracy_after_averaging": test_accuracy})

    return test_accuracy