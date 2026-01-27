from dataloader.dataloader import classification_dataset
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description='Load configuration file')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    config_path = args.config

    config = None
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = classification_dataset(root_dir=config['folder'])

if __name__ == '__main__':
    main()