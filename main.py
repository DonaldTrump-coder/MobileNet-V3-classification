from dataloader.dataloader import classification_dataset
import yaml
import argparse
from models.mobilenetV3 import small_MobileNetV3
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Load configuration file')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    config_path = args.config

    config = None
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    transform = transforms.Compose([
        transforms.Resize((config['data']['resize']['height'], config['data']['resize']['width'])),
        transforms.ToTensor(),
    ])
    
    dataset = classification_dataset(root_dir=config['data']['folder'], transform=transform)
    labels = dataset.labels
    train_indices, test_indices = train_test_split(
        np.arange(len(dataset)), 
        train_size=config['data']['train_size'], 
        stratify=labels
    )
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    for i, (inputs, labels) in enumerate(train_loader):
        print(inputs.shape, labels.shape)
        break

if __name__ == '__main__':
    main()