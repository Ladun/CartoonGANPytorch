
import os
import argparse
import logging

import torch
from torchvision import transforms

from model.model import Generator
from utils.utils import load_image_dataloader, save, load
from config import Config

logger = logging.getLogger(__name__)

def load_generator(args, config):

    if not os.path.exists(args.weights):
        logger.info(f"Generator weights {args.weights} does not exist!")
        return None

    generator = Generator(config)
    generator_state_dict = torch.load(args.weights, map_location=args.device)
    generator.load_state_dict(generator_state_dict)
    generator.to(args.device)

    return generator

def generate_and_save(args, generator, image_loader):
    
    logger.info('Testing...')
    generator.eval()

    torch_to_image = transforms.Compose([
        transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),  # [-1, 1] to [0, 1]
        transforms.ToPILImage()
    ])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_ix = 0
    for test_images, _ in image_loader:
        test_images = test_images.to(args.device)
        generated_images = generator(test_images).detach().cpu()

        for i in range(len(generated_images)):
            image = generated_images[i]
            image = torch_to_image(image)
            image.save(os.path.join(args.output_dir, '{0}.jpg'.format(image_ix)))
            image_ix += 1

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="output/ckpt/generator.pt",
                        help='path to generator weights')
    parser.add_argument("--model_config", type=str, default="model_config.json")    
    parser.add_argument('--input', type=str, default="data/input",
                        help='path to input images')    
    parser.add_argument('--output_dir', type=str, default="output/test",
                        help='path to output images')    

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    args = parser.parse_args()
    
    # Setup logging    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    model_config = Config.load(args.model_config)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    generator = load_generator(args, model_config)
    if generator is None:
        exit()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    image_loader, _ = load_image_dataloader(args.input, transform)
    generate_and_save(args, generator, image_loader)

if __name__ == "__main__":
    main()