# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
import pickle
from PIL import Image, ImageDraw

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T

import scipy.stats as stats

class UECFoodNewPixDataset(object):
    def __init__(self, split, transforms):

        self.imgroot = "/export/space0/okamoto-ka/dataset/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/{}/img/".format(split)
        self.pixroot = "/export/space0/okamoto-ka/dataset/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/{}/mask/".format(split)
        self.newbbroot = "/export/space0/okamoto-ka/dataset/new_uecfood100/BB/"

        self.uecfood100bb = "/export/space0/okamoto-ka/dataset/uecfood100BB/{}/".format(split)

        self.transforms = transforms
        self.picklepath = "./file_number_BB_{}.pickle".format(split)

        # load all image files, sorting them to
        # ensure that they are aligned
        
        with open(self.picklepath,'rb') as f:
            self.file_numbers = pickle.load(f)
            if split == 'train':
                self.file_numbers = self.file_numbers

    def __getitem__(self, idx):
        
        file_number = self.file_numbers[idx]
        img_path = os.path.join(self.imgroot,(file_number + '.jpg'))
        mask_path = os.path.join(self.pixroot,(file_number + '.png'))
        bb_txt = os.path.join(self.newbbroot,(file_number + '.txt'))
        # load images ad masks
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).convert("RGB")

        if not (os.path.exists(img_path) and os.path.exists(mask_path)):
            print(img_path)
        
        mask,_,_ = mask.split()
        
        # split the color-encoded mask into a set
        # of binary masks
        # get bounding box coordinates for each mask
        boxes = []

        with open(bb_txt,'r') as f:
            lines = f.read()
            lines = lines.split("\n")

            if lines[-1] == "":          
                lines.pop()        

        for line in lines:
            pos = line.split()
            if int(pos[0]) == 1:
                continue

            pos = pos[1:]

            xmin = self.abs2two(int(pos[0]))
            xmax = self.abs2two(int(pos[2]))
            ymin = self.abs2two(int(pos[1]))
            ymax = self.abs2two(int(pos[3]))
            boxes.append([xmin, ymin, xmax, ymax])
    
        obj_ids = []
        masks = np.zeros((len(boxes),mask.size[1],mask.size[0]))
        #draw image 
        draw = ImageDraw.Draw(img)
        save_boxes = []
        
        for i,b in enumerate(boxes):
            
            mini_mask = np.array(mask.crop((b[0],b[1],b[2],b[3])))
            count = np.unique(mini_mask)
            count = count.tolist()
            mask_id = 0
            mask_id_count = 0

            for c in count:                
                if c == 0:
                    continue
                else:
                    id_count = np.sum(mini_mask[mini_mask == c]) / c
                
                    if id_count > mask_id_count:
                        mask_id = c
                        mask_id_count = id_count
                                    
            #draw img
            #draw.rectangle((int(b[0]),int(b[1]),int(b[2]),int(b[3])),outline=(0,255,0))
            #draw.text((int(b[0]),int(b[1])),str(mask_id))

            #del mask_id 101 and 102
            obj_ids.append(mask_id)
            if (mask_id == 101 or mask_id == 102):
                pass
            else:
                bbline = "{} {} {} {} {}\n".format(str(2),b[0],b[1],b[2],b[3])
                save_boxes.append(bbline)

            # 0 ga itibannno tokiwa kouryo si
            instance_mini_mask = np.where(mini_mask == c,1,0)
            masks[i,b[1]:b[3],b[0]:b[2]] = instance_mini_mask

            #instance_mini_mask = instance_mini_mask.astype(np.uint8)
            #pil_instance = Image.fromarray(instance_mini_mask)
            #pil_instance.save('./demo_mask/{}_{}'.format(i,os.path.basename(img_path)),quality = 95)

        with open(os.path.join(self.uecfood100bb,(file_number + '.txt')),'w') as f:
            
            f.writelines(save_boxes)

        #img.save('./demo_img/{}'.format(os.path.basename(img_path)),quality = 95)

        num_objs = len(obj_ids)
        obj_ids = np.asarray(obj_ids)
        #masks = mask == obj_ids[:,None,None]

        labels = torch.as_tensor((obj_ids),dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)        

        return img, target

    def __len__(self):
        return len(self.file_numbers)

    def abs2two(self, x):
        if x < 2 :
            return 2
        else:
            return x


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


dataset = UECFoodNewPixDataset('train',None)
#dataset = UECFoodNewPixDataset('test',None)

#data_num = len(dataset)

for i,d in enumerate(dataset):
    
    if i % 1000 == 0:
        print(i)
    '''
    target = d[1]
    boxes = target['boxes']
    label = target['labels']
    masks = target['masks']
    
    print(boxes)
    print(label)
    print(masks)
    
    break
    '''
    
