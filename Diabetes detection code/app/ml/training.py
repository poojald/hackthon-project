import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
import json
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.history = None
        self.metrics = {}
        
    def train(self, train_data, val_data, epochs=50, callbacks=None):
        """
        Train the model
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of training epochs
            callbacks: List of Keras callbacks
        
        Returns:
            Training history
        """
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_data, save_dir='reports'):
        """
        Comprehensive model evaluation
        
        Args:
            test_data: Test dataset
            save_dir: Directory to save evaluation reports
        
        Returns:
            Dictionary of metrics
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Get predictions
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for images, labels in test_data:
            predictions = self.model.predict(images, verbose=0)
            y_pred_proba.extend(predictions)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        
        # Calculate metrics
        self.metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Generate visualizations
        self.plot_confusion_matrix(y_true, y_pred, save_dir)
        self.plot_roc_curves(y_true, y_pred_proba, save_dir)
        self.plot_precision_recall_curves(y_true, y_pred_proba, save_dir)
        
        if self.history:
            self.plot_training_history(save_dir)
        
        # Save metrics report
        self.save_metrics_report(save_dir)
        
        return self.metrics
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }
        
        # ROC AUC
        try:
            y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
            metrics['auc_roc_macro'] = float(roc_auc_score(y_true_bin, y_pred_proba, average='macro'))
            metrics['auc_roc_weighted'] = float(roc_auc_score(y_true_bin, y_pred_proba, average='weighted'))
        except:
            metrics['auc_roc_macro'] = 0.0
            metrics['auc_roc_weighted'] = 0.0
        
        # Confusion matrix values
        cm = confusion_matrix(y_true, y_pred)
        
        # For binary classification or calculate for each class
        if len(self.class_names) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positive'] = int(tp)
            metrics['true_negative'] = int(tn)
            metrics['false_positive'] = int(fp)
            metrics['false_negative'] = int(fn)
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        else:
            # Calculate specificity for each class
            specificities = []
            for i in range(len(self.class_names)):
                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp = np.sum(cm[:, i]) - cm[i, i]
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificities.append(spec)
            metrics['specificity_macro'] = float(np.mean(specificities))
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_dir):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also plot normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'confusion_matrix_normalized.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, y_true, y_pred_proba, save_dir):
        """Plot ROC curves for each class"""
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i in range(len(self.class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Class', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curves(self, y_true, y_pred_proba, save_dir):
        """Plot Precision-Recall curves"""
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(len(self.class_names)):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            
            plt.plot(recall, precision, lw=2,
                    label=f'{self.class_names[i]} (AP = {avg_precision:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'precision_recall_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self, save_dir):
        """Plot training history"""
        if not self.history:
            return
        
        history_dict = self.history.history
        
        # Plot accuracy
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history_dict['accuracy'], label='Training Accuracy')
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy', fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history_dict['loss'], label='Training Loss')
        plt.plot(history_dict['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss', fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics_report(self, save_dir):
        """Save metrics to JSON file"""
        report_path = os.path.join(save_dir, 'metrics_report.json')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Also save as text
        text_path = os.path.join(save_dir, 'metrics_report.txt')
        with open(text_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DIABETES DETECTION MODEL - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {report['timestamp']}\n\n")
            
            f.write("Overall Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Accuracy: {self.metrics['accuracy']:.4f}\n")
            f.write(f"Precision (Macro): {self.metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro): {self.metrics['recall_macro']:.4f}\n")
            f.write(f"F1-Score (Macro): {self.metrics['f1_macro']:.4f}\n")
            f.write(f"AUC-ROC (Macro): {self.metrics.get('auc_roc_macro', 0):.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            f.write("-" * 60 + "\n")
            for class_name, class_metrics in self.metrics['per_class'].items():
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_metrics['f1_score']:.4f}\n")


def perform_cross_validation(model_builder, X, y, n_splits=5, epochs=30):
    """
    Perform k-fold cross-validation
    
    Args:
        model_builder: Function that returns a compiled model
        X: Input data
        y: Labels
        n_splits: Number of folds
        epochs: Training epochs per fold
    
    Returns:
        Dictionary with cross-validation results
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_scores = []
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")
        print("-" * 50)
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build new model for this fold
        model = model_builder()
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            verbose=0
        )
        
        # Evaluate
        score = model.evaluate(X_val, y_val, verbose=0)
        cv_scores.append(score[1])  # Accuracy
        fold_histories.append(history.history)
        
        print(f"Fold {fold + 1} Accuracy: {score[1]:.4f}")
    
    results = {
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'fold_scores': cv_scores,
        'fold_histories': fold_histories
    }
    
    print(f"\nCross-Validation Results:")
    print(f"Mean Accuracy: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")
    
    return results
