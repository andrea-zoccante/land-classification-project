# libraries

import os
# import numpy as np
import pandas as pd
import joblib
# import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import random

# from glob import glob
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import CLIPProcessor, CLIPModel

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
# import seaborn as sns

from tqdm import tqdm
from config import CLASSES, MOD_CLASSES
from imageDatasets import EuroSATDataset
from classifiers import MLPClassifier, LinearClassifier
from PromptLearner import SimplePromptLearner, TextEncoder


class customCLIP:
    def __init__(self, 
                 model_name = "openai/clip-vit-base-patch32", 
                 full_prompt=True, 
                 modify=True,
                 augment_hue=False):
        self.clip_model = CLIPModel.from_pretrained(model_name)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.device = (
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )
        self.clip_model.to(self.device)

        self.image_dataset = EuroSATDataset("2750/")

        self.full_prompt = full_prompt
        if modify:
            self.class_labels = MOD_CLASSES
        else:
            self.class_labels = CLASSES
        self.augment_hue = augment_hue

        val_df = pd.read_csv("validation_data/validation_set.csv")

        self.val_labels = torch.tensor([CLASSES.index(cls) for cls in val_df["Class"].tolist()], dtype=torch.long).to(self.device)
        self.val_image_paths = val_df["Image Path"].tolist()

        self.classifier = None
        self.prompt_learner = None

        self.testing_mode = "zeroshot"

        pass

    def set_testing_mode(self, mode):
        mode = mode.lower()
        valid_modes = ["zeroshot", "probe", "coop"]

        if mode not in valid_modes:
            raise ValueError(f"{mode} not recognized. Must be one of: {valid_modes}")
        print(f"Testing in {mode} mode")
        self.testing_mode = mode

    def load_model(self, model_path, mode):
        valid_modes = {"linear_probe": LinearClassifier, "mlp_probe": MLPClassifier, "coop": None, "logreg_probe": None}
        if mode not in valid_modes:
            raise ValueError(f"{mode} not recognized. Must be one of: {valid_modes}")

        if os.path.exists(model_path):
            if mode in ["linear_probe", "mlp_probe"]:
                classifier = valid_modes[mode]
                self.classifier = classifier(input_dim=512, num_classes=len(CLASSES)).to(self.device)  # Adjust `input_dim` if needed
                self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            elif mode == "logreg_probe":
                self.classifier = joblib.load(model_path)
            elif mode == "coop":
                prompt_learner = SimplePromptLearner(self.clip_model, self.class_labels, n_ctx=16).to(self.device)
                prompt_learner.load_state_dict(torch.load(model_path, map_location=self.device))
                self.prompt_learner = prompt_learner
            
            print(f"Loaded {mode} model from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")


    def process_images(self, image_paths, text_inputs=None, coop=False):
        images = []
    
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            if self.augment_hue:
                # Apply hue augmentation
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))
            
            images.append(image)
        
        if text_inputs is None:
            text_inputs = [f"A satellite image of {cls}" for cls in self.class_labels] if self.full_prompt else self.class_labels

        if not coop:
            inputs = self.clip_processor(text=text_inputs, images=images, return_tensors="pt", padding=True).to(self.device)
        else:
            inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
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
                    raise ValueError("No trained classifier found. Train using train() first.")

                # Extract image features
                image_features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"]).to(self.device)

                if isinstance(self.classifier, LogisticRegression):  
                    predictions = self.classifier.predict(image_features.cpu().numpy())  # Convert tensor to NumPy
                    return torch.tensor(predictions, dtype=torch.long, device=self.device)  # Convert back to tensor

                # Apply trained classifier
                logits = self.classifier(image_features)
                return logits.argmax(dim=-1)
            
            elif self.testing_mode == "coop":
                if self.prompt_learner is None:
                    raise ValueError("No trained promptLearner found. Train using train() first.")
                
                prompts = self.prompt_learner()
                text_encoder = TextEncoder(clip_model=self.clip_model).to(self.device)

                text_features = text_encoder(prompts)
                image_features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"]).to(self.device)
                
                logit_scale = self.clip_model.logit_scale.exp()
                test_logits = image_features @ text_features.T
                test_logits *= logit_scale
                
                predictions = torch.argmax(test_logits, dim=1)
                return torch.tensor(predictions, dtype=torch.long, device=self.device)

    
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
        print(f"Testing dataset using {self.testing_mode} method")
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
        print(f"\nTraining {mode.replace('_', ' ')} with {few_shot}-shot data...")

        mode = mode.lower()
        valid_modes = {"linear_probe": LinearClassifier, "mlp_probe": MLPClassifier, "logreg_probe": None, "coop": None} #, "logreg_probe": LogisticRegression
        
        if mode not in valid_modes:
            raise ValueError(f"{mode} not recognized. Must be one of: {list(valid_modes.keys())}")

        few_shot_df = pd.read_csv(f"few_shot_data/few_shot_{few_shot}.csv")
        if mode == "coop":
            few_shot_image_inputs = self.process_images(few_shot_df["Image Path"].tolist(), coop=True)
        else:
            few_shot_image_inputs = self.process_images(few_shot_df["Image Path"].tolist(), coop=False)
            few_shot_image_features = self.extract_image_features_with_text(few_shot_df["Image Path"].tolist())
        
        few_shot_labels = torch.tensor([CLASSES.index(cls) for cls in few_shot_df["Class"]], dtype=torch.long).to(self.device)        

        if mode in ["linear_probe", "mlp_probe"]:
            classifier = valid_modes[mode](input_dim=few_shot_image_features.shape[1], num_classes=len(CLASSES)).to(self.device)
            results = self.train_model(model=classifier, 
                                       mode=mode,
                                       labels=few_shot_labels,
                                       image_features=few_shot_image_features
                                       )
            # results = self.train_LinearProbe(classifier, few_shot_image_features, few_shot_labels, val_image_features)
            self.classifier = results["model"]

            if save_model:
                save_path = f"models/classifiers/{mode}/{few_shot}-shot.pth"
                torch.save(self.classifier.state_dict(), save_path)
                print(f"{mode} model saved at {save_path}")
        elif mode == "coop":
            prompt_learner = SimplePromptLearner(self.clip_model, self.class_labels, n_ctx=16).to(self.device)
            text_encoder = TextEncoder(clip_model=self.clip_model).to(self.device)
            results = self.train_model(model=prompt_learner, 
                                       mode=mode,
                                       labels=few_shot_labels,
                                       text_encoder=text_encoder,
                                       inputs=few_shot_image_inputs
                                       )
            self.prompt_learner = results["model"]
            if save_model:
                save_path = f"models/prompt_learners/{mode}/{few_shot}-shot.pth"
                torch.save(self.prompt_learner.state_dict(), save_path)
                print(f"{mode} model saved at {save_path}")
            
        elif mode == "logreg_probe":
            log_reg = LogisticRegression(max_iter=1000)
            log_reg.fit(few_shot_image_features.cpu().numpy(), few_shot_labels.cpu().numpy())
            self.classifier = log_reg

            if save_model:
                save_path = f"models/classifiers/{mode}/{few_shot}-shot.pkl"
                joblib.dump(log_reg, save_path)
                print(f"Logistic Regression model saved at {save_path}")

        # prompts = prompt_learner()
        
        # # Get updated text features using learned prompts, and image features
        # text_features = text_encoder(prompts)
        # image_features = clip_model.get_image_features(**inputs)

        # # Normalize features
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # # Compute cosine similarity
        # logit_scale = clip_model.logit_scale.exp()
        # logits = image_features @ text_features.T
        # logits *= logit_scale

    def train_model(self, model, mode, labels, image_features=None, text_encoder=None, inputs=None):
        lr = {"coop": 0.01, "linear_probe": 0.001, "mlp_probe": 0.001}[mode]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        losses = []
        
        for epoch in range(1001):
            optimizer.zero_grad()
            if mode in ["linear_probe", "mlp_probe"]:
                logits = model(image_features.to(self.device))
            elif mode == "coop":
                logits = self.get_logits_CoOp(model, text_encoder, inputs)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())

            if epoch % 250 == 0:  
                print(f"Epoch [{epoch}/{1000}] | Train Loss: {loss.item():.4f} | ")
        results = {
            "model": model,
            "losses": losses,
        }
        return results

    def get_logits_CoOp(self, prompt_learner, text_encoder, inputs):
        prompts = prompt_learner()
        
        # Get updated text features using learned prompts, and image features
        text_features = text_encoder(prompts)
        image_features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])

        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        logit_scale = self.clip_model.logit_scale.exp()
        logits = image_features @ text_features.T
        logits *= logit_scale
        return logits

    
    # def train_LinearProbe(self, classifier, image_features, labels, val_image_features):
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        
    #     losses = []
    #     val_losses = []
    #     val_accuracies = []
        
    #     for epoch in range(1001):
    #         # classifier.train()
    #         optimizer.zero_grad()
            
    #         logits = classifier(image_features.to(self.device))
    #         loss = criterion(logits, labels)
    #         loss.backward()
    #         optimizer.step()
            
    #         losses.append(loss.item())
            
    #         if epoch % 50 == 0:
    #             with torch.no_grad():
    #                 val_logits = classifier(val_image_features.to(self.device))
    #                 val_loss = criterion(val_logits, self.val_labels)
                    
    #                 predictions = torch.argmax(val_logits, dim=1)
                
    #             accuracy = (predictions == self.val_labels).float().mean().item()
                
    #             val_accuracies.append(accuracy)
    #             val_losses.append(val_loss.item())
                
    #             if epoch % 250 == 0:  
    #                 print(f"Epoch [{epoch}/{1000}] | Train Loss: {loss.item():.4f} | "
    #                     f"Val Loss: {val_loss.item():.4f} | Val Acc: {accuracy * 100:.2f}%")
    
        
    #     results = {
    #         "classifier": classifier,
    #         "losses": losses,
    #         "val_losses": val_losses,
    #         "val_accuracies": val_accuracies
    #     }
    #     return results
# customclip = customCLIP(model_name="openai/clip-vit-base-patch32", full_prompt=True, modify=True,augment_hue=True)
# results = customclip.train(few_shot=8, mode="CoOp", save_model=True)