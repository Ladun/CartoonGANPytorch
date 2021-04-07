import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

def perform_edge_smoothing(root_dir, save_dir, kernel_size=5, canny_threshold1=50, canny_threshold2=250):
    """
    Perform edge smoothing to given image, then save.

    :param root_dir: directory containing photo imagesas
    :param save_dir: path to save an image
    :kernel_size: kernel size to be used for edge dilation and Gaussian kernel. Must be an odd number.
    :canny_threshold1: first threshold for cv2.Canny
    :canny_threshold2: second threshold for cv2.Canny
    """
    assert kernel_size % 2 == 1  # kernel size must be odd
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_paths = glob.glob(os.path.join(root_dir, '*'))

    for image_path in tqdm(image_paths):
        if os.path.isdir(image_path):
            # directory. ignore
            continue

        bgr_img = cv2.imread(image_path)
        gray_img = cv2.imread(image_path, 0)
        cl1 = clahe.apply(gray_img)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        gauss = cv2.getGaussianKernel(kernel_size, 0)
        gauss = gauss * gauss.transpose(1, 0)

        padding = (kernel_size - 1) // 2
        pad_img = np.pad(bgr_img, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')

        edges = cv2.Canny(cl1, canny_threshold1, canny_threshold2)  # detect edges
        dilation = cv2.dilate(edges, kernel)  # dilate edges

        smoothed_img = np.copy(bgr_img)
        idx = np.where(dilation != 0)  # index of dilation where value is not 0 == index near edges

        for i in range(len(idx[0])):
            # index of dilation where value is not 0 means index is near edges
            # change these points value using convolution between gauss kernel and are near the points
            # causing these points (and thus image) to become less sharp and blurry
            smoothed_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(
                        pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            smoothed_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(
                        pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            smoothed_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(
                        pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

            # here, pad_img[i:i+kernel, j:j+kernel] == bgr_img[i-pad:i-pad+kernel, j-pad:j-pad+kernel]

        image_name = (image_path.split('/')[-1]).split('\\')[-1]  # ex. images/1.jpg -> 1.jpg
        cv2.imwrite(os.path.join(save_dir, image_name), smoothed_img)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--root_path',
                        help='Path to directory of animation images to be preprocessed. ')

    parser.add_argument('--save_path',
                        help='Path to directory where preprocessed animation images will be saved. ')

    parser.add_argument('--target_size',
                        type=int,
                        default=256,
                        help='target size of preprocessed images')


    args = parser.parse_args()

    if args.root_path is None or args.save_path is None:
        parser.error('--root_path and --save_path is required.')
    elif not os.path.isdir(args.root_path) or not os.path.isdir(args.save_path) :
        parser.error('--root_path and --save_path must be existing directories.')
    else:
        perform_edge_smoothing(args.root_path, args.save_path)

if __name__ == '__main__':
    main()
