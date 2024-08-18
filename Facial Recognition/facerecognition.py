import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import clip
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import faiss
import numpy as np

# CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

athletes = ["Mbappe", "Kohli", "Lebron", "Messi", "Ronaldo"]
image_folder_path = '/Users/akshaypozath/Facial Recognition/images'


image_paths = [os.path.join(image_folder_path, fname) for fname in ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg", "11.jpg"]]

# text descriptions of athletes
text_descriptions = [f"a photo of {name}" for name in athletes]
text_inputs = clip.tokenize(text_descriptions).to(device)


with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# feature extraction function
def extract(image_paths, model, preprocess, device):
    image_features_list = []
    for image_path in image_paths:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features_list.append(image_features.cpu())
    return image_features_list

# store image features
image_features_list = extract(image_paths, model, preprocess, device)

#convert to np arrays
image_features_np= np.vstack([features.numpy() for features in image_features_list])
text_features_np= text_features.cpu().numpy()

#faiss index
dim= 512
index=faiss.IndexFlatL2(dim)
#adding text features
index.add(text_features_np)

# query the faiss index with image features
k = 1  
distances, indices = index.search(image_features_np, k)



# accuracy and prediction
predictions = predictions = [athletes[idx] for idx in indices.flatten()]

for image_path, prediction in zip(image_paths, predictions):
    print(f"Image: {image_path} - Predicted celebrity: {prediction}") 
