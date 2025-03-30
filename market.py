from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch.utils.data as data
from glob import glob

class market(data.Dataset):
    def __init__(self, root, domain, train=True, transform=None, from_file=False):
        self.train = train
        self.transform = transform
        
        if domain == 'source':
            domain = 'train'
        else:
            if self.train:
                domain = 'validation'
            else:
                domain = 'test'

        if train:
          folder = 'bounding_box_train'
        else:
          folder = 'bounding_box_test'

        if not from_file:
            data = []
            labels = []
            address = f"{root}/{folder}/*.jpg"
            files = glob(address)

            image_labels = []
            for item in files:
              if os.path.basename(item).split('_')[0] not in image_labels:
                image_labels.append(os.path.basename(item).split('_')[0])

            # Create a unique mapping
            name_to_int = {name: idx for idx, name in enumerate(set(image_labels))}
            int_to_name = {idx: name for name, idx in name_to_int.items()}

            for j , item in enumerate(files):
              data.append(item)
              labels.append(name_to_int[os.path.basename(item).split('_')[0]])

            np.random.seed(1234)
            idx = np.random.permutation(len(data))

            self.data = np.array(data)[idx]
            self.labels = np.array(labels)[idx]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.labels[index]          

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img)
         
        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.data)
