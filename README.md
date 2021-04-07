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
    python train.py


# Test
    python infer.py --image=path/to/image or path/to/image_dir --output=path/to/save