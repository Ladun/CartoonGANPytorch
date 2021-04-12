# CartoonGAN-pytorch


# Prepare data


Alternatively you can build your own dataset by setting up the following directory structure:

    .
    ├── data                   
    |   ├── edge_smooth_image      
    |   |   ├── image              # Training
    |   |   |   ├── A              # Contains domain edge smoothed images
    |   ├── target      
    |   |   ├── image              # Training
    |   |   |   ├── A              # Contains domain target style images
    |   ├── photo      
    |   |   ├── image              # Training
    |   |   |   ├── A              # Contains domain real world images

# Train
    python train.py --checkpoint_dir=path/to/ckpt --model_config=path/to/model_config


# Test


following directory structure:  

    .
    ├── input_dir                   
    |   ├── image              # Training
    |   |   ├── A              # Contains domain edge smoothed images
    |   |   ├── ...            # Contains domain edge smoothed images

## Testing code

    python infer.py --weights=path/to/generator.pt --model_config=path/to/model_config --input=path/to/input --output_dir=path/to/output_dir