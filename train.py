import argparse
import logging
import random

import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
from tensorboardX import SummaryWriter

import torch.nn as nn
import torch.optim
from torchvision import transforms

from config import Config
from model.model import Generator, Discriminator, FeatureExtractor
from utils.utils import load_image_dataloader, save, load

logger = logging.getLogger(__name__)

def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)


def train(args, 
          generator : Generator, discriminator: Discriminator, feature_extractor: FeatureExtractor,
          photo_dataloader, edge_smooth_dataloader, animation_dataloader,
          checkpoint_dir=None):

    tb_writter = SummaryWriter()

    gen_criterion = nn.BCELoss().to(args.device)
    disc_criterion = nn.BCELoss().to(args.device)
    content_criterion = nn.L1Loss().to(args.device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.adam_beta, 0.999))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.adam_beta, 0.999))

    global_step = 0
    global_init_step = 0

    # The number of steps to skip when loading a checkpoint
    skipped_step = 0
    skipped_init_step = 0

    cur_epoch = 0
    cur_init_epoch = 0

    data_len = min(len(photo_dataloader), len(edge_smooth_dataloader), len(animation_dataloader))
    
    if checkpoint_dir:
        try:
            checkpoint_dict = load(checkpoint_dir)
            generator.load_state_dict(checkpoint_dict['generator'])
            discriminator.load_state_dict(checkpoint_dict['discriminator'])
            gen_optimizer.load_state_dict(checkpoint_dict['gen_optimizer'])
            disc_optimizer.load_state_dict(checkpoint_dict['disc_optimizer'])
            global_step = checkpoint_dict['global_step']
            global_init_step = checkpoint_dict['global_init_step']

            cur_epoch = global_step // data_len
            cur_init_epoch = global_init_step // len(photo_dataloader)

            skipped_step = global_step % data_len
            skipped_init_step = global_init_step % len(photo_dataloader)

            logger.info("Start training with,")
            logger.info("In initialization step, epoch: %d, step: %d", cur_init_epoch, skipped_init_step)
            logger.info("In main train step, epoch: %d, step: %d", cur_epoch, skipped_step)
        except:
            logger.info("Wrong checkpoint path")

    t_total =  data_len * args.n_epochs
    t_init_total = len(photo_dataloader) * args.n_init_epoch

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num photo examples = %d", len(photo_dataloader))
    logger.info("  Num edge_smooth examples = %d", len(edge_smooth_dataloader))
    logger.info("  Num animation examples = %d", len(animation_dataloader))

    logger.info("  Num Epochs = %d", args.n_epochs)
    logger.info("  Total train batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    logger.info("  Num Init Epochs = %d", args.n_init_epoch)
    logger.info("  Total Init optimization steps = %d", t_init_total)

    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)


    init_phase = True
    try:
        generator.train()
        discriminator.train()

        gloabl_init_loss = 0
        # --- Initialization Content loss
        mb = master_bar(range(cur_init_epoch, args.n_init_epoch))
        for init_epoch in mb:
            epoch_iter = progress_bar(photo_dataloader, parent=mb)
            for step, (photo, _)  in enumerate(epoch_iter):
                if skipped_init_step > 0:
                    skipped_init_step =- 1
                    continue

                photo = photo.to(args.device)

                gen_optimizer.zero_grad()
                x_features = feature_extractor((photo + 1) / 2).detach()
                Gx = generator(photo)
                Gx_features = feature_extractor((Gx + 1) / 2)

                content_loss = args.content_loss_weight * content_criterion(Gx_features, x_features)
                content_loss.backward()
                gen_optimizer.step()

                gloabl_init_loss += content_loss.item()

                global_init_step += 1

                if args.save_steps > 0 and global_init_step % args.save_steps == 0:
                    logger.info("Save Initialization Phase, init_epoch: %d, init_step: %d", init_epoch, global_init_step)
                    save(checkpoint_dir, global_step, global_init_step, generator, discriminator, gen_optimizer, disc_optimizer)

                if args.logging_steps > 0 and global_init_step % args.logging_steps == 0:
                    tb_writter.add_scalar('Initialization Phase/Content Loss', content_loss.item(), global_init_step)   
                    tb_writter.add_scalar('Initialization Phase/Global Generator Loss', gloabl_init_loss / global_init_step, global_init_step)   
                    
                    logger.info("Initialization Phase, Epoch: %d, Global Step: %d, Content Loss: %.4f", init_epoch, global_init_step, gloabl_init_loss / (global_init_step))        

        # -----------------------------------------------------
        logger.info("Finish Initialization Phase, save model...")
        save(checkpoint_dir, global_step, global_init_step, generator, discriminator, gen_optimizer, disc_optimizer)

        init_phase = False
        global_loss_D = 0
        global_loss_G = 0
        global_loss_content = 0

        mb = master_bar(range(cur_epoch, args.n_epochs))
        for epoch in mb:
            epoch_iter = progress_bar(list(zip(animation_dataloader, edge_smooth_dataloader, photo_dataloader)), parent=mb)
            for step, ((animation, _), (edge_smoothed, _), (photo, _))  in enumerate(epoch_iter):
                if skipped_step > 0:
                    skipped_step =- 1
                    continue

                animation = animation.to(args.device)
                edge_smoothed = edge_smoothed.to(args.device)
                photo = photo.to(args.device)

                disc_optimizer.zero_grad()
                # --- Train discriminator
                # ------ Train Discriminator with animation image
                animation_disc = discriminator(animation)
                animation_target = torch.ones_like(animation_disc)
                loss_animation_disc = disc_criterion(animation_disc, animation_target)

                # ------ Train Discriminator with edge image
                edge_smoothed_disc = discriminator(edge_smoothed)
                edge_smoothed_target = torch.zeros_like(edge_smoothed_disc)
                loss_edge_disc = disc_criterion(edge_smoothed_disc, edge_smoothed_target)

                # ------ Train Discriminator with generated image
                generated_image = generator(photo).detach()
                
                generated_image_disc = discriminator(generated_image)
                generated_image_target = torch.zeros_like(generated_image_disc)
                loss_generated_disc = disc_criterion(generated_image_disc, generated_image_target)

                loss_disc = loss_animation_disc + loss_edge_disc + loss_generated_disc

                loss_disc.backward()
                disc_optimizer.step()

                global_loss_D += loss_disc.item()

                # --- Train Generator
                gen_optimizer.zero_grad()

                generated_image = generator(photo)

                generated_image_disc = discriminator(generated_image)
                generated_image_target = torch.ones_like(generated_image_disc)
                loss_adv = gen_criterion(generated_image_disc, generated_image_target)

                # ------ Train Generator with content loss
                x_features = feature_extractor((photo + 1) / 2).detach()
                Gx_features =feature_extractor((generated_image + 1) /2 )

                loss_content = args.content_loss_weight * content_criterion(Gx_features, x_features)

                loss_gen = loss_adv + loss_content
                loss_gen.backward()
                gen_optimizer.step()

                global_loss_G += loss_adv.item()
                global_loss_content += loss_content.item()

                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    logger.info("Save Training Phase, epoch: %d, step: %d", epoch, global_step)
                    save(checkpoint_dir, global_step, global_init_step, generator, discriminator, gen_optimizer, disc_optimizer)

                if args.logging_steps > 0 and global_init_step % args.logging_steps == 0:     
                    tb_writter.add_scalar('Train Phase/Generator Loss', loss_adv.item(), global_step)   
                    tb_writter.add_scalar('Train Phase/Discriminator Loss', loss_disc.item(), global_step)      
                    tb_writter.add_scalar('Train Phase/Content Loss', loss_content.item(), global_step)   
                    tb_writter.add_scalar('Train Phase/Global Generator Loss', global_loss_G / global_step, global_step)   
                    tb_writter.add_scalar('Train Phase/Global Discriminator Loss', global_loss_D / global_step, global_step)      
                    tb_writter.add_scalar('Train Phase/Global Content Loss', global_loss_content / global_step, global_step)    

                    logger.info("Training Phase, Epoch: %d, Global Step: %d, Disc Loss %.4f, Gen Loss %.4f, Content Loss: %.4f", 
                                epoch, global_step, global_loss_D / global_step, global_loss_G / global_step, global_loss_content / global_step)

    except KeyboardInterrupt:
        
        if init_phase:
            logger.info("KeyboardInterrupt in Initialization Phase!")
            logger.info("Save models, init_epoch: %d, init_step: %d", init_epoch, global_init_step)
        else:
            logger.info("KeyboardInterrupt in Training Phase!")
            logger.info("Save models, epoch: %d, step: %d", epoch, global_step)

        save(checkpoint_dir, global_step, global_init_step, generator, discriminator, gen_optimizer, disc_optimizer)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_dir", type=str, default="output/ckpt")
    parser.add_argument("--model_config", type=str, default="model_config.json")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
                        
    parser.add_argument('--photo_dir', type=str, default="data/photo",
                        help='path to photo datasets.')    
    parser.add_argument('--edge_smooth_dir', type=str, default="data/edge_smooth",
                        help='path to edge_smooth datasets.')    
    parser.add_argument('--target_dir', type=str, default="data/target",
                        help='path to target datasets.')   
        
    parser.add_argument('--content_loss_weight', type=float, default=10, 
                        help='content loss weight')
    parser.add_argument('--seed', type=int, default=42, 
                        help='seed')
    parser.add_argument('--adam_beta', type=float, default=0.5, 
                        help='adam_beta')
    parser.add_argument('--n_epochs', type=int, default=100, 
                        help='number of epochs of training')
    parser.add_argument('--n_init_epoch', type=int, default=15, 
                        help='number of epochs of initializing')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, 
                        help='initial learning rate')
    parser.add_argument('--n_cpu', type=int, default=0, 
                        help='number of cpu threads to use during batch generation')   
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")                 
    parser.add_argument('--save_steps', type=int, default=3000, 
                        help='Save checkpoint every X updates steps.')


    args = parser.parse_args()

    # Setup logging    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    model_config = Config.load(args.model_config)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    logger.warning("device: %s, n_gpu: %s",
                    args.device, args.n_gpu)

    
    generator = Generator(model_config).to(args.device)
    discriminator = Discriminator(model_config).to(args.device)
    feature_extractor = FeatureExtractor(model_config).to(args.device)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    photo_dataloader, _ = load_image_dataloader(args.photo_dir, transform, args.batch_size, args.n_cpu)
    edge_smooth_dataloader, _ = load_image_dataloader(args.edge_smooth_dir, transform, args.batch_size, args.n_cpu)
    animation_dataloader, _ = load_image_dataloader(args.target_dir, transform, args.batch_size, args.n_cpu)

    train(args, 
          generator, discriminator, feature_extractor, 
          photo_dataloader, edge_smooth_dataloader, animation_dataloader,
          args.checkpoint_dir)


if __name__ == "__main__":
    main()