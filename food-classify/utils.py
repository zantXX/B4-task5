import matplotlib.pyplot as plt
import matplotlib
import torch
from torchvision import datasets, transforms
import os
import numpy as np
import datetime
#from pudb import set_trace
from os.path import join

def display_losses(train_losses, val_losses, title, folder='plots'):
    matplotlib.use('Agg')
    if not os.path.exists(folder):
        os.makedirs(folder)

    x_axis = np.arange(len(train_losses))
    fig, ax = plt.subplots()
    ax.axis(ymin=0., ymax=1.2)  # hard coded to benefit comparisions


    ax.plot(x_axis, train_losses, 'r-', label='train')
    ax.plot(x_axis, val_losses, 'b-', label='val')
    ax.legend()

    ax.set(xlabel='Epochs', ylabel='Epoch Loss', title=title)
    ax.grid()

    fig.savefig(join(folder, title + '.png'))



def show_batch(inp, title=None, folder='plots'):
    """Imshow for Tensor."""
    matplotlib.use('Agg')
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    #plt.imshow(inp)
    if title is not None:
        plt.title(title)
        plt.savefig(join(folder, 'batch_FOOD101.png'))


def get_data_loaders(data_dir, batch_size):
    # For training, augment and normalize images
    # For validation/test, just normalize images
    data_transforms = {
	'train': transforms.Compose([
	    transforms.RandomResizedCrop(224),
	    transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406],
				 std=[0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
	    transforms.Resize(256),
	    transforms.CenterCrop(224),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406],
				 std=[0.229, 0.224, 0.225])
	]),
	'test': transforms.Compose([
	    transforms.Resize(256),
	    transforms.CenterCrop(224),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406],
				 std=[0.229, 0.224, 0.225])
	]),
    }
    
    image_datasets = {x: datasets.ImageFolder(data_dir,data_transforms[x])
		      for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
						  shuffle=True, num_workers=4)
		   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names

# Use of TenCrop
# https://pytorch.org/docs/master/torchvision/transforms.html#torchvision.transforms.TenCrop
# transform = Compose([
#    TenCrop(size), # this is a list of PIL Images
#    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
# ])
# #In your test loop you can do the following:
# input, target = batch # input is a 5d tensor, target is 2d
# bs, ncrops, c, h, w = input.size()
# result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
# result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops

def get_tencrop_data_loader(data_dir, batch_size):
    #  experimental
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = {
	'test': transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224), # this is a list of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop)
                                                         for crop in crops])), # returns a 4D tensor
            transforms.Lambda(lambda crops: torch.stack([normalize(crop)
                                                         for crop in crops])),
	]),
    }
    
    image_datasets = {x: datasets.ImageFolder(data_dir, data_transforms[x])
		      for x in ['test']}
    set_trace()
    dataloader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
						  shuffle=True, num_workers=4)
		   for x in ['test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
    class_names = image_datasets['test'].classes
    return dataloader, dataset_sizes, class_names



def get_logfilename_with_datetime(prefix):
    # Use current date/time (upto minute) to get a text file name.
    return prefix + "-" + \
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + \
        ".log"
