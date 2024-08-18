import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image, ImageTk
import clip
import os
import faiss
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Canvas, Scrollbar, Frame

# Set environment variables to avoid potential conflicts
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# List of athlete names
athletes = ["Mbappe", "Kohli", "Lebron", "Messi", "Ronaldo"]

# Create text descriptions of athletes
text_descriptions = [f"a photo of {name}" for name in athletes]
text_inputs = clip.tokenize(text_descriptions).to(device)

# Encode text descriptions
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_features_np = text_features.cpu().numpy()

# Create FAISS index
dim = 512
index = faiss.IndexFlatL2(dim)
index.add(text_features_np)

# Function to extract image features
def extract(image_paths, model, preprocess, device):
    image_features_list = []
    for image_path in image_paths:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features_list.append(image_features.cpu())
    return image_features_list

# Function to run the prediction
def run_prediction(image_folder_path):
    try:
        image_paths = [os.path.join(image_folder_path, fname) for fname in os.listdir(image_folder_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        if not image_paths:
            messagebox.showerror("Error", "No images found in the selected folder.")
            return
        
        image_features_list = extract(image_paths, model, preprocess, device)
        image_features_np = np.vstack([features.numpy() for features in image_features_list])

        k = 1  # Number of nearest neighbors to search
        distances, indices = index.search(image_features_np, k)

        predictions = [athletes[idx] for idx in indices.flatten()]

        display_results(image_paths, predictions)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to display the results
def display_results(image_paths, predictions):
    result_window = tk.Toplevel(root)
    result_window.title("Prediction Results")

    canvas = Canvas(result_window)
    scrollbar = Scrollbar(result_window, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    for i, (image_path, prediction) in enumerate(zip(image_paths, predictions)):
        img = Image.open(image_path)
        img.thumbnail((150, 150))
        img = ImageTk.PhotoImage(img)

        panel = tk.Label(scrollable_frame, image=img)
        panel.image = img  # Keep a reference to avoid garbage collection
        panel.grid(row=i, column=0, padx=10, pady=10)

        label = tk.Label(scrollable_frame, text=f"Predicted: {prediction}")
        label.grid(row=i, column=1, padx=10, pady=10)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

# Function to select folder
def select_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        run_prediction(folder_selected)

# Create the main window
root = tk.Tk()
root.title("Celebrity Image Recognition")

# Add a button to select the folder
btn_select_folder = tk.Button(root, text="Select Image Folder", command=select_folder)
btn_select_folder.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()