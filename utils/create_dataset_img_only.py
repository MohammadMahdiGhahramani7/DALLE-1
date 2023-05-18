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
  
def Create_Dataset(img_path, img_size):
  
  DATASET = []

  for img_name in tqdm(os.listdir(img_path)):

    full_name = img_path + "/" + img_name
    
    img = Image.open(full_name)

    img_data = img_preprocessing(img, img_size)

    DATASET.append(img_data)
      
  return DATASET


img_path = sys.argv[1]
img_size = int(sys.argv[2])

dataset = Create_Dataset(img_path, img_size)

with open(f'{img_path}_{img_size}_img_only.pkl', 'wb') as f:
      
    pickle.dump(dataset, f)
