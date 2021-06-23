import os
import numpy as np
import pickle
from PIL import Image, ImageDraw
import cv2

def putfile(split = 'train'):

 imgpath = "/export/space0/okamoto-ka/dataset/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/{}/img/".format(split)
 pixpath = "/export/space0/okamoto-ka/dataset/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/{}/mask/".format(split)
 newbbpath = "/export/space0/okamoto-ka/dataset/new_uecfood100/BB/"    
 uecfood100bb = "/export/space0/okamoto-ka/dataset/uecfood100BB/{}/".format(split)

 masklist = os.listdir(pixpath)
 file_numbers = []

 bblist = os.listdir(uecfood100bb)
 print(split,len(bblist))


 for i, b in enumerate(bblist):
  
  if '~' in b:
   pass
  else :
   
   bpath = os.path.join(uecfood100bb,b)

   with open(bpath) as f:
    lines = f.read()
    lines = lines.split('\n')
       
   if lines == ['']:
     print(bpath)

   else:
     pix_name_number = b.replace('.txt','') + '.png'
     img_name_number = b.replace('.txt','') + '.jpg'

     pixfile = os.path.join(pixpath,pix_name_number)
     imgfile = os.path.join(imgpath,img_name_number)

     if (os.path.exists(pixfile) and os.path.exists(imgfile)) :
    
      file_numbers.append(b.replace('.txt',''))

     with open('file_number_BB100_{}.pickle'.format(split),'wb') as f:
      pickle.dump(file_numbers,f)
      

  if i % 1000 == 0:
   print(i)

def loadimg(split = 'train'):

 imgroot = "/export/space0/okamoto-ka/dataset/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/{}/img/".format(split)
 pixroot = "/export/space0/okamoto-ka/dataset/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/{}/mask/".format(split)
 newbbroot = "/export/space0/okamoto-ka/dataset/new_uecfood100/BB/"

 picklepath = "./file_number_{}.pickle".format(split)
 # load all image files, sorting them to
 # ensure that they are aligned

 with open(picklepath,'rb') as f:
  file_numbers = pickle.load(f)
  

 file_number = file_numbers[5]
 img_path = os.path.join(imgroot,(file_number + '.jpg'))
 mask_path = os.path.join(pixroot,(file_number + '.png'))
 bb_txt = os.path.join(newbbroot,(file_number + '.txt'))
 
 
 with open(bb_txt,'r') as f:
  lines = f.read()
  lines = lines.split("\n")

  if lines[-1] == "":
   lines.pop()

 boxes = []

 print(lines)
 for line in lines:

  pos = line.split()
  pos = pos[1:]
  xmin = int(pos[0])
  xmax = int(pos[2])
  ymin = int(pos[1])
  ymax = int(pos[3])
  boxes.append([(xmin,ymin),(xmax,ymax)])
  
 
  img = cv2.imread(img_path,1)
 
 for b in boxes:
  
  cv2.rectangle(img,b[0],b[1],(255, 0, 0),thickness=8)

 cv2.imwrite(file_number+'.png',img)

#loadimg()

putfile('train')
putfile('test')
