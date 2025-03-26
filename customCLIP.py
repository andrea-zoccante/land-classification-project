# libraries

import os
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

        self.classifier = None

        self.testing_mode = "zeroshot"

        pass

    def load_classifier(self, classifier_path):
        valid_modes = {"linear_probe": LinearClassifier, "mlp_probe": MLPClassifier}

        if os.path.exists(classifier_path):
            # Extract the mode from the classifier path, e.g., "models/classifiers/Linear_Probe/8-shot"
            classifier_parts = classifier_path.split(os.sep)
            
            # The mode should be the second-to-last part of the path (before the number of shots)
            mode = classifier_parts[-2].lower()  # Get 'Linear_Probe' or 'MLP_Probe'

            if mode not in valid_modes:
                raise ValueError(f"Unknown classifier mode: {mode}. Valid modes are {list(valid_modes.keys())}.")

            # Load the correct classifier model based on the extracted mode
            self.classifier = valid_modes[mode](input_dim=512, num_classes=len(CLASSES)).to(self.device)  # Adjust `input_dim` if needed
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
            print(f"Loaded {mode} classifier from {classifier_path}")
        else:
            raise FileNotFoundError(f"Classifier file not found at {classifier_path}")


    def process_images(self, image_paths, text_inputs=None):
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        
        if text_inputs is None:
            text_inputs = [f"A satellite image of {cls}" for cls in self.class_labels] if self.full_prompt else self.class_labels

        inputs = self.clip_processor(text=text_inputs, images=images, return_tensors="pt", padding=True).to(self.device)
        
        return inputs
    
    def classify_images_clip(self, image_paths):
        """
        Classifies images using CLIP, either in a zero-shot manner or with a trained classifier.

        Parameters:
        - image_paths (list of str): Paths to the images.
        - mode (str): Classification mode, either 'zeroshot' or 'linear_probe' (trained classifier).
        
        Returns:
        - Tensor: Predicted class indices.
        """

        inputs = self.process_images(image_paths)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move to GPU if available

        with torch.no_grad():
            if self.testing_mode == "zeroshot":
                # Zero-shot classification using CLIP similarity scores
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                return probs.argmax(dim=-1)

            elif self.testing_mode == "probe":
                if self.classifier is None:
                    raise ValueError("No trained classifier found. Train using train_LinearProbe first.")

                # Extract image features
                image_features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"]).to(self.device)

                # Apply trained classifier
                logits = self.classifier(image_features)
                return logits.argmax(dim=-1)

    
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
    
    def multi_class_analysis(self, mode):
        mode = mode.lower()
        valid_modes = ["zeroshot", "probe"]

        if mode not in valid_modes:
            raise ValueError(f"{mode} not recognized. Must be one of: {valid_modes}")
        
        self.testing_mode = mode
        
        results = []

        for eval_class in CLASSES:
            class_results = self.single_class_analysis(eval_class=eval_class)
            
            results.append(class_results)

        results_df = pd.DataFrame(results)

        return results_df 
    
    def extract_image_features_with_text(self, image_paths):
        inputs = self.process_images(image_paths)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        return image_features.cpu()

    
    def train(self, few_shot, mode: str, save_model: bool=False):
        mode = mode.lower()
        valid_modes = {"linear_probe": LinearClassifier, "mlp_probe": MLPClassifier} #, "logreg_probe": LogisticRegression
        
        if mode not in valid_modes:
            raise ValueError(f"{mode} not recognized. Must be one of: {list(valid_modes.keys())}")

        val_image_features = self.extract_image_features_with_text(self.val_image_paths)
        few_shot_df = pd.read_csv(f"few_shot_data/few_shot_{few_shot}.csv")
        few_shot_image_features = self.extract_image_features_with_text(few_shot_df["Image Path"].tolist())
        few_shot_labels = torch.tensor([CLASSES.index(cls) for cls in few_shot_df["Class"]], dtype=torch.long).to(self.device)

        classifier = valid_modes[mode](input_dim=few_shot_image_features.shape[1], num_classes=len(CLASSES)).to(self.device)

        print(f"\nTraining {mode.replace('_', ' ')} with {few_shot}-shot data...")

        if mode in ["linear_probe", "mlp_probe"]:
            results = self.train_LinearProbe(classifier, few_shot_image_features, few_shot_labels, val_image_features)
            classifier = results["classifier"]
            if save_model:
                save_path = f"models/classifiers/{mode}/{few_shot}-shot.pth"
                torch.save(classifier.state_dict(), save_path)
                print(f"Model saved at {save_path}")


    
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
                    
                    predictions = torch.argmax(val_logits, dim=1)
                
                accuracy = (predictions == self.val_labels).float().mean().item()
                
                val_accuracies.append(accuracy)
                val_losses.append(val_loss.item())
                
                if epoch % 250 == 0:  
                    print(f"Epoch [{epoch}/{1000}] | Train Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss.item():.4f} | Val Acc: {accuracy * 100:.2f}%")
    
        
        results = {
            "classifier": classifier,
            "losses": losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies
        }
        return results
        