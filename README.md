# Data Augmentation - Mixed Images

This code generates mixed images by concatenating two half images obtained from each image and save the images in a new parent folder './data/mixedimages'. 


## Data Setting

Basically, you should organize your train directory to use torchvision.datasets.ImageFolder.

+ Custom dataset - use the following data structure (characteristic for PyTorch):
    ```
    -data
      -train
        -class1
            -image_1
            -image_2
            -...
        -class2
            -image_1
            -image_2
            -...
        -...
      -test
        -class1
            -image_1
            -image_2
            -...
        -class2
            -image_1
            -image_2
            -...
        -...
    ```

## Code Example
The train and mixedimage sets are gathered by torch.utils.data.ConcatDataset.
```python
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
        #print(imgs)
        print(imgs.size(), labels.size())
        exit(0)
```
## Image Sample
<img width="458" alt="Screen Shot 2021-01-18 at 9 14 10 AM" src="https://user-images.githubusercontent.com/52735725/104888840-991c2d00-596d-11eb-9792-f1c74e82deff.png">
