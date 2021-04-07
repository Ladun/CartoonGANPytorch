
import torch

from model.model import FeatureExtractor, Generator, Discriminator
from config import Config

from utils.utils import load_image_dataloader

def main():
    # class TestDataset:

    #     def __init__(self, r):
            
    #         self.data = range(r)

    #     def __getitem__(self, idx):
    #         return self.data[idx];

    # a = TestDataset(7)
    # b = TestDataset(4)
    # c = TestDataset(9)

    # for _ in range(3):
    #     for _a, _b, _c in zip(a, b, c):
    #         print(_a, _b, _c)


    dataloader, dataset = load_image_dataloader('data/photo', 8, 4)


    i = 0

    for d, l in dataloader:
        
        print(f"{d.size()}  {l.size()}")

        i += 1

        if i > 5:
            break

if __name__ == "__main__":
    main()