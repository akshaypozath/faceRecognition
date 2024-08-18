# faceRecognition
Facial Recognition: Create a facial recognition algorithm that utilizes CLIP model (Contrastive Learning Image Pretraining) in order to isolate features given an image and make predictions given those features


CLIP Model: The clip model enables the user to identify the association between both text and images using the image and text encoder that the CLIP model contains


STEPS TO CREATE ALGORITHM: 

-Data Collection: Collected different facial images with the use case being athletes

-Image Folder Selection: User is prompted to select any image folder of their choosing

-Embeddings: CLIP model creates both textual (name of athelete) and image embeddings

-Connection to Vector DB(Faiss): Allows the algorithm to handle much larger datasets as well as finding an efficient and precise prediction

-Nearest Neighbor Search: Utilize Faiss' IndexFlatL2 to conduct nearest neighbor search ensuring the the predictions are precise and accurate. 

-Prediction: Prediction is made given the results of Nearest Neighbor Search

*I created a simple scrollable UI using Tkinter to allow users to efficiently verify the results of the Search*
