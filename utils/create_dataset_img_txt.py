import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import sys
import pickle

def img_preprocessing(img, img_size):

  data_transforms = [
      transforms.Resize((img_size, img_size)), #Resize all images
      transforms.ToTensor() # scale between [0, 1]
  ]

  data_transform = transforms.Compose(data_transforms)

  return data_transform(img)

def Create_Dataset(img_path, cap_path, img_size):

  with open(cap_path, 'r') as f:

    lines = f.readlines()

    captions = [x.strip().split('.jpg,') for x in lines][1:]

    f.close()
  
  DATASET = []

  for img_name in tqdm(os.listdir(img_path)):

    full_name = img_path + "/" + img_name
    
    img = Image.open(full_name)

    img_data = img_preprocessing(img, img_size)

    for corr in list(filter(lambda x:x[0]==img_name[:-4], captions)):#(corr_caps := list(filter(lambda x:x[0]==img_name[:-4], captions))):
      #corr : [name, txt]

      #assert len(corr_caps) == 5, "ERROR" #for each image we have 5 alternative txts
      assert len(corr[1]) > 1, f"ERROR {corr}" #len(txt) must be greater than 1
      
      DATASET.append([img_data, corr])
      
  return DATASET


img_path = sys.argv[1]
cap_path = sys.argv[2]
img_size = int(sys.argv[3])

dataset = Create_Dataset(img_path, cap_path, img_size)

with open(f'{img_path}_{img_size}.pkl', 'wb') as f:
      
    pickle.dump(dataset, f)
    
    
