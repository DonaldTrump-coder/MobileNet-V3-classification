from dataloader.dataloader import classification_dataset
import yaml
import argparse
from models.mobilenetV3 import small_MobileNetV3, large_MobileNetV3
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import logging
import os
from tqdm import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter
import subprocess

log_file = 'model_info.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def main():
    os.makedirs('output', exist_ok=True)
    if os.path.exists(log_file):
        open(log_file, 'w').close()
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

    model = None
    if config['model'] == 'MobileNetV3_small':
        model = small_MobileNetV3(num_classes=dataset.get_num_classes())
    elif config['model'] == 'MobileNetV3_large':
        model = large_MobileNetV3(num_classes=dataset.get_num_classes())
    else:
        raise ValueError('Unsupported model type: ' + config['model'])
    logging.info('Using Model: ' + config['model'])

    if os.path.exists(os.path.join('output', config['model'])):
        if os.path.isdir(os.path.join('output', config['model'])):
            shutil.rmtree(os.path.join('output', config['model']))
        else:
            os.remove(os.path.join('output', config['model']))
    os.makedirs(os.path.join('output', config['model']), exist_ok=True)
    output_dir = os.path.join('output', config['model'])
    tensor_board_writer = SummaryWriter(log_dir=output_dir)
    tensor_board_process = subprocess.Popen(['tensorboard', f'--logdir={output_dir}'])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total trainable parameters: {total_params}')

    # train
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.NAdam(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    num_epochs = config['epochs']
    for epoch in range(num_epochs):
        training_loss = 0.0
        total = 0
        correct = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for i, (inputs, labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    model = model.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                tensor_board_writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)
                tensor_board_writer.add_scalar('Training Accuracy', 100 * correct / total, epoch * len(train_loader) + i)
            
                pbar.set_postfix({'loss': training_loss / (i + 1), 'accuracy': 100 * correct / total})
                pbar.update(1)
        
        tensor_board_writer.add_scalar('Training epoch Loss', training_loss/len(train_loader), epoch)

        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {training_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

    tensor_board_writer.close()
    logging.info('Training complete.')

    # inference
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
                model = model.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logging.info(f'Test Accuracy: {100 * correct / total:.2f}%')

    tensor_board_process.terminate()

if __name__ == '__main__':
    main()