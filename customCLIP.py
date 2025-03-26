# libraries

# import os
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from PIL import Image
# from glob import glob
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import CLIPProcessor, CLIPModel

from sklearn.metrics import confusion_matrix, accuracy_score
# import seaborn as sns

from tqdm import tqdm
from config import CLASSES, MOD_CLASSES
from imageDatasets import EuroSATDataset
from classifiers import MLPClassifier, LinearClassifier

class customCLIP:
    def __init__(self, 
                 model_name = "openai/clip-vit-base-patch32", 
                 full_prompt=True, 
                 modify=False):
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.device = (
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )
        self.clip_model.to(self.device)

        self.image_dataset = EuroSATDataset("2750/")

        self.full_prompt = full_prompt
        # self.modify = modify
        if modify:
            self.class_labels = MOD_CLASSES
        else:
            self.class_labels = CLASSES

        val_df = pd.read_csv("validation_data/validation_set.csv")

        self.val_labels = torch.tensor([CLASSES.index(cls) for cls in val_df["Class"].tolist()], dtype=torch.long).to(self.device)
        self.val_image_paths = val_df["Image Path"].tolist()

        # self.mode = training_mode

        pass
    def classify_images_clip(self, image_paths):
        """
        Classifies a single image using CLIP in a zero-shot manner.

        Parameters:
        - image_path (str): Path to the image.
        - full_prompt (bool): Include context in text input
        - modify (bool): Use modified class names
        """

        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

        text_inputs = [f"A satellite image of {cls}" for cls in self.class_labels] if self.full_prompt else self.class_labels

        inputs = self.clip_processor(text=text_inputs, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move to GPU if available

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        return probs.argmax(dim=-1)
    
    def single_class_analysis(self, eval_class):
        class_subset = self.image_dataset.get_class_subset(eval_class)
        class_dataloader = DataLoader(class_subset, batch_size=64, shuffle=False)

        y_true = []
        y_pred = []


        for img_paths, labels in tqdm(class_dataloader, desc=f"Classifying {eval_class} batches: ", leave=True):
            y_true.extend(labels.cpu().numpy())

            preds = self.classify_images_clip(image_paths=img_paths)
            y_pred.extend(preds.cpu().numpy())

        results = {
            "class": eval_class,
            "y_true": y_true,
            "y_pred": y_pred,
            "accuracy": accuracy_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true,y_pred, labels=list(range(10)))
        }
        return results
    
    def multi_class_analysis(self):
        results = []

        for eval_class in CLASSES:
            class_results = self.single_class_analysis(eval_class=eval_class)
            
            results.append(class_results)

        results_df = pd.DataFrame(results)

        return results_df 
    
    def extract_image_features_with_text(self, image_paths):
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        text_inputs = [f"A satellite image of {cls}" for cls in self.class_labels]

        inputs = self.clip_processor(text=text_inputs, images=images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        
        return image_features.cpu()
    
    def train(self, few_shot, mode: str):
        if mode.lower() not in ["linear_probe", "mlp_probe", "logteg_probe", "coop"]:
            raise ValueError(f"{mode} not recognized, must be one of: [linear_probe, mlp_probe, logteg_probe, coop]")
        val_image_features = self.extract_image_features_with_text(self.val_image_paths)

        few_shot_df = pd.read_csv(f"few_shot_data/few_shot_{few_shot}.csv")
        few_shot_image_features = self.extract_image_features_with_text(few_shot_df["Image Path"].tolist())
        few_shot_labels = torch.tensor([CLASSES.index(cls) for cls in few_shot_df["Class"]], dtype=torch.long).to(self.device)

        if mode == "Linear_Probe" or mode == "MLP_Probe":
            classifier_model = {"Linear_Probe": LinearClassifier, "MLP_Probe": MLPClassifier}[mode]
            classifier = classifier_model(input_dim=few_shot_image_features.shape[1], num_classes=len(CLASSES)).to(self.device)

            print(f"\nLinear-Probe ({mode}) finetuning with {few_shot} images per class...")
            self.train_LinearProbe(classifier=classifier, 
                                   image_features=few_shot_image_features,
                                   labels=few_shot_labels,
                                   val_image_features=val_image_features)
    
    def train_LinearProbe(self, classifier, image_features, labels, val_image_features):

        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        
        losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(1001):
            classifier.train()
            optimizer.zero_grad()
            
            logits = classifier(image_features.to(self.device))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 50 == 0:
                with torch.no_grad():
                    val_logits = classifier(val_image_features.to(self.device))
                    val_loss = criterion(val_logits, self.val_labels)

                    val_loss = criterion(val_logits, self.val_labels)
                    
                    predictions = torch.argmax(val_logits, dim=1)
                
                accuracy = (predictions == self.val_labels).float().mean().item()
                
                val_accuracies.append(accuracy)
                val_losses.append(val_loss.item())
                
                if epoch % 250 == 0:  
                    print(f"Epoch [{epoch}/{1000}] | Train Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss.item():.4f} | Val Acc: {accuracy * 100:.2f}%")
    
        
        results = {
            "model": classifier,
            "losses": losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }
        return results
        
# customclip = customCLIP(model_name="openai/clip-vit-base-patch32", full_prompt=True, modify=True)
# results = customclip.train(few_shot=8, mode="Linear_Probe")
# # image_path = "2750/Forest/Forest_100.jpg"
# # best_class_index = customclip.classify_images_clip([image_path])
# # print(f"Predicted Class: {CLASSES[best_class_index]}")
# true, pred = customclip.single_class_analysis("Forest")
# print(accuracy_score(true, pred))