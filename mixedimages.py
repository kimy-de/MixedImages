import numpy as np
from PIL import Image
import os
import random
import glob

def newfolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')

def mixedimageloader(trainpath, num_generated_images=2, image_size=128):

    # Read train image paths
    image_paths = sorted(glob.glob(trainpath+'/*/*'))

    # Extract Labels
    labels = []
    for path in image_paths:
        labels.append(path.split('/')[-2])
    label_group = sorted(list(set(labels)))
    labels = np.array(labels)

    # Create mixed images
    parent_folder = './data/mixedimages'
    newfolder(parent_folder)
    half = image_size//2
    for i in range(len(label_group)):
        newfolder(parent_folder+'/'+label_group[i])
        idx = list(np.where(labels == label_group[i])[0])

        for j in range(num_generated_images):
            sampled_list = random.sample(idx, 2)
            first = image_paths[sampled_list[0]]
            second = image_paths[sampled_list[1]]
            img1 = np.array(Image.open(first).resize((image_size, image_size)))
            img2 = np.array(Image.open(second).resize((image_size, image_size)))

            if len(img1.shape) > 1:
                img1[half:, :, :] = img2[half:, :, :]
            else:
                img1[half:, :] = img2[half:, :]

            im = Image.fromarray(img1)
            im.save(parent_folder+'/'+label_group[i]+'/' + label_group[i] + '_' + str(j) + '.png')

# Code Tester
if __name__ == "__main__":


    import torch
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    Generator = False
    if Generator == True:
        mixedimageloader('./data/train', 10, 80)

    trans = transforms.Compose([transforms.Resize((80,80)), transforms.ToTensor()])
    dataset = ImageFolder(root='./data/train', transform=trans)
    mixedbook = ImageFolder(root='./data/mixedimages', transform=trans)
    l = []
    l.append(dataset)
    l.append(mixedbook)
    concatdataset = torch.utils.data.ConcatDataset(l)

    trainloader = DataLoader(concatdataset, batch_size=30, shuffle=True)

    for imgs, labels in trainloader:
        print(imgs.size(), labels.size())
        exit(0)
