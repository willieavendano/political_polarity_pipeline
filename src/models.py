"""
Machine learning models for political polarity classification.

Includes baseline (Logistic Regression) and transformer (DistilBERT) models.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

logger = logging.getLogger(__name__)


class BaselineClassifier:
    """Baseline classifier using TF-IDF + Logistic Regression or SVM."""
    
    def __init__(
        self,
        classifier_type: str = "logistic",
        C: float = 1.0,
        class_weight: str = "balanced",
        calibration: Optional[str] = "isotonic",
        random_state: int = 42,
    ):
        """
        Initialize baseline classifier.
        
        Args:
            classifier_type: "logistic" or "svm"
            C: Regularization parameter
            class_weight: Class weighting strategy
            calibration: Calibration method ("isotonic", "sigmoid", or None)
            random_state: Random seed
        """
        self.classifier_type = classifier_type
        self.calibration = calibration
        self.random_state = random_state
        
        # Initialize base classifier
        if classifier_type == "logistic":
            self.base_model = LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=1000,
                random_state=random_state,
                n_jobs=-1,
            )
        elif classifier_type == "svm":
            self.base_model = LinearSVC(
                C=C,
                class_weight=class_weight,
                max_iter=1000,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        self.model = None
        self.num_classes = None
    
    def train(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: Optional[csr_matrix] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaselineClassifier":
        """
        Train the classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for calibration)
            y_val: Validation labels (for calibration)
            
        Returns:
            Self
        """
        logger.info(f"Training {self.classifier_type} classifier...")
        logger.info(f"Training samples: {X_train.shape[0]}, features: {X_train.shape[1]}")
        
        self.num_classes = len(np.unique(y_train))
        
        # Train base model
        self.base_model.fit(X_train, y_train)
        
        # Apply calibration if requested
        if self.calibration and X_val is not None and y_val is not None:
            logger.info(f"Applying {self.calibration} calibration...")
            self.model = CalibratedClassifierCV(
                self.base_model,
                method=self.calibration,
                cv="prefit",
            )
            self.model.fit(X_val, y_val)
        else:
            self.model = self.base_model
        
        # Log training accuracy
        train_acc = self.model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_acc:.4f}")
        
        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_acc:.4f}")
        
        return self
    
    def predict(self, X: csr_matrix) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        """Predict class probabilities."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # SVM without calibration doesn't have predict_proba
            decision = self.model.decision_function(X)
            # Simple softmax
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
    
    def get_feature_importance(
        self,
        feature_names: List[str],
        top_n: int = 30,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top features (coefficients) for each class.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features per class
            
        Returns:
            Dictionary mapping class to list of (feature, coefficient) tuples
        """
        # Get coefficients
        if self.calibration:
            # For calibrated models, extract base estimator coefficients
            base_estimator = self.model.calibrated_classifiers_[0].estimator
            if hasattr(base_estimator, "coef_"):
                coef = base_estimator.coef_
            else:
                logger.warning("Model does not have coefficients")
                return {}
        else:
            if hasattr(self.model, "coef_"):
                coef = self.model.coef_
            else:
                logger.warning("Model does not have coefficients")
                return {}
        
        # Extract top features per class
        class_features = {}
        
        for class_idx in range(coef.shape[0]):
            class_coef = coef[class_idx]
            top_indices = np.argsort(np.abs(class_coef))[::-1][:top_n]
            top_features = [
                (feature_names[idx], class_coef[idx])
                for idx in top_indices
            ]
            class_features[class_idx] = top_features
        
        return class_features


class TransformerClassifier:
    """Transformer-based classifier using pretrained language models."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 3,
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        """
        Initialize transformer classifier.
        
        Args:
            model_name: Pretrained model name from HuggingFace
            num_classes: Number of output classes
            max_length: Maximum sequence length
            device: Device to use ("cuda", "cpu", or None for auto-detect)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
        )
        self.model.to(self.device)
        
        self.trainer = None
    
    def prepare_dataset(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
    ) -> Dataset:
        """
        Prepare dataset for training/inference.
        
        Args:
            texts: List of text strings
            labels: Optional list of labels
            
        Returns:
            HuggingFace Dataset
        """
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        
        # Create dataset
        data_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
        
        if labels is not None:
            data_dict["labels"] = labels
        
        return Dataset.from_dict(data_dict)
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 500,
        early_stopping_patience: int = 3,
        fp16: bool = True,
    ) -> "TransformerClassifier":
        """
        Train the transformer model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            output_dir: Directory to save model checkpoints
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Learning rate warmup steps
            logging_steps: Logging frequency
            eval_steps: Evaluation frequency
            save_steps: Checkpoint save frequency
            early_stopping_patience: Patience for early stopping
            fp16: Whether to use mixed precision (requires GPU)
            
        Returns:
            Self
        """
        logger.info("Preparing datasets...")
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        val_dataset = self.prepare_dataset(val_texts, val_labels)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=fp16 and self.device == "cuda",
            report_to="none",
            seed=42,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )
        
        # Train
        logger.info("Starting training...")
        self.trainer.train()
        logger.info("Training completed!")
        
        return self
    
    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for inference
            
        Returns:
            Array of predicted labels
        """
        probas = self.predict_proba(texts, batch_size)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for inference
            
        Returns:
            Array of class probabilities
        """
        dataset = self.prepare_dataset(texts)
        
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_data = dataset[i:i+batch_size]
                
                input_ids = torch.tensor(batch_data["input_ids"]).to(self.device)
                attention_mask = torch.tensor(batch_data["attention_mask"]).to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.vstack(all_probs)
    
    def save(self, output_dir: str):
        """Save model and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def load(cls, model_path: str, device: Optional[str] = None) -> "TransformerClassifier":
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model directory
            device: Device to load model on
            
        Returns:
            Loaded TransformerClassifier instance
        """
        logger.info(f"Loading model from {model_path}...")
        
        # Detect number of classes from config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        num_classes = config.num_labels
        
        # Create instance
        instance = cls(
            model_name=model_path,
            num_classes=num_classes,
            device=device,
        )
        
        logger.info("Model loaded successfully")
        return instance
