
import os
import json

import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model.model import Generator, Discriminator



def load_image_dataloader(root_dir, transform, batch_size=1, num_workers=0, shuffle=True):
    """
    :param root_dir: directory that contains another directory of images. All images should be under root_dir/<some_dir>/
    :param batch_size: batch size
    :param num_workers: number of workers for torch.utils.data.DataLoader
    :param shuffle: use shuffle
    :return: torch.utils.Dataloader object
    """
    assert os.path.isdir(root_dir)

    image_dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    dataloader = DataLoader(image_dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_workers)

    return dataloader, image_dataset

def save(ckpt_dir_path, global_step, global_init_step,
        generator:Generator, discriminator:Discriminator,
        gen_optimizer, disc_optimizer):   

    if not os.path.exists(ckpt_dir_path):
        os.makedirs(ckpt_dir_path)

    torch.save(generator.state_dict(), os.path.join(ckpt_dir_path, "generator.pt"))
    torch.save(discriminator.state_dict(), os.path.join(ckpt_dir_path, "discriminator.pt"))
    torch.save(gen_optimizer.state_dict(), os.path.join(ckpt_dir_path, "generator_optimizer.pt"))
    torch.save(disc_optimizer.state_dict(), os.path.join(ckpt_dir_path, "discriminator_optimizer.pt"))
    
    with open(os.path.join(ckpt_dir_path, "learning_state.json"), 'w') as f:
        json.dump({      
            'global_step': global_step,
            'global_init_step': global_init_step,
        }, f, indent='\t')

def load(ckpt_dir_path):

    if not os.path.exists(ckpt_dir_path):
        return None

    generator_state_dict = torch.load(os.path.join(ckpt_dir_path, "generator.pt"))
    discriminator_state_dict = torch.load(os.path.join(ckpt_dir_path, "discriminator.pt"))
    generator_optimizer_state_dict = torch.load(os.path.join(ckpt_dir_path, "generator_optimizer.pt"))
    discriminator_optimizer_state_dict = torch.load(os.path.join(ckpt_dir_path, "discriminator_optimizer.pt"))

    with open(os.path.join(ckpt_dir_path, "learning_state.json"), 'r') as f:
        state = json.load(f)    

    return {
        'generator': generator_state_dict,
        'discriminator': discriminator_state_dict,
        'gen_optimizer': generator_optimizer_state_dict,
        'disc_optimizer': discriminator_optimizer_state_dict,
        'global_step': state['global_step'],
        'global_init_step': state['global_init_step']
    }
