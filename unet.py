import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Suppress oneDNN messages
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class FraudDetectionUNet:
    def __init__(self, hidden_layers=(64, 32, 16), learning_rate=0.001):
        self.hidden_layers = hidden_layers
        self.model = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.learning_rate = learning_rate

    def create_unet_model(self, input_dim):
        """Builds a U-Net architecture for tabular fraud detection"""
        inputs = Input(shape=(input_dim,))
        encoder_layers = []
        x = inputs

        # Encoder Path
        for units in self.hidden_layers:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            encoder_layers.append(x)

        # Bottleneck
        bottleneck = Dense(self.hidden_layers[-1] // 2, activation='relu')(x)

        # Decoder Path with Skip Connections
        for i, units in enumerate(reversed(self.hidden_layers)):
            x = Dense(units, activation='relu')(bottleneck if i == 0 else x)
            x = BatchNormalization()(x)
            x = Concatenate()([x, encoder_layers[-(i + 1)]])

        # Output Layer
        outputs = Dense(input_dim, activation='tanh')(x)

        # Compile model
        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

    def fit(self, X_train_normal, X_val, y_val, epochs=50, batch_size=32, validation_split=0.1):
        """Trains the U-Net model on normal transactions with validation set"""
        X_scaled = self.scaler.fit_transform(X_train_normal)

        # Train only on normal transactions
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        # Compute reconstruction error on validation set (contains fraud cases)
        X_val_scaled = self.scaler.transform(X_val)
        reconstructed_val = self.model.predict(X_val_scaled)
        reconstruction_error_val = np.mean(np.power(X_val_scaled - reconstructed_val, 2), axis=1)

        # Find optimal threshold using validation set with fraud cases
        precision, recall, thresholds = precision_recall_curve(y_val, reconstruction_error_val)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        
        # Handle edge case where no valid threshold found
        if best_idx < len(thresholds):
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = np.percentile(reconstruction_error_val, 95)
            
        self.threshold = best_threshold

        print(f"Optimal Threshold Selected: {self.threshold:.6f}")

        # Freeze layers
        for layer in self.model.layers:
            layer.trainable = False

        # Plot training history
        self.plot_training_history(history)

        return history

    def plot_training_history(self, history):
        """Plots the training loss vs. validation loss"""
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.show()

    def predict(self, X):
        """Predict fraud likelihood based on reconstruction error"""
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        return (mse > self.threshold).astype(int)

    def get_reconstruction_error(self, X):
        """Compute reconstruction error for a given dataset"""
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled)
        return np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

    def evaluate_model(self, X, y_true):
        """Evaluate model performance metrics"""
        reconstruction_error = self.get_reconstruction_error(X)

        # Predictions and metrics
        y_pred = (reconstruction_error > self.threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        # Compute Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Compute TPR and FPR using Confusion Matrix
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate

        # Compute AUC Score
        fpr_vals, tpr_vals, _ = roc_curve(y_true, reconstruction_error)
        roc_auc = auc(fpr_vals, tpr_vals)

        # Print Metrics
        print(f"AUC Score: {roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"TPR (Recall): {tpr:.4f}")
        print(f"FPR: {fpr:.4f}")

        # Plot Confusion Matrix
        self.plot_confusion_matrix(cm)

        # Plot ROC Curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr_vals, tpr_vals, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self, cm):
        """Plots the confusion matrix"""
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Fraud"],
                    yticklabels=["Normal", "Fraud"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

def load_and_preprocess_data(filepath):
    """Loads and prepares the credit card transaction data"""
    df = pd.read_csv(filepath)
    
    print(f"\nDataset Info:")
    print(f"Total Transactions: {len(df)}")
    print(f"Fraud Cases: {df['Class'].sum()}")
    print(f"Fraud Percentage: {df['Class'].mean()*100:.2f}%")

    X = df.drop('Class', axis=1)
    y = df['Class']

    # Initial split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

def main():
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('creditcard.csv')

    # Create validation set with fraud cases
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Apply SMOTE to validation set (not training set, to preserve anomaly detection paradigm)
    smote = SMOTE(random_state=42)
    X_val_smote, y_val_smote = smote.fit_resample(X_val, y_val)
    print(f"\nAfter SMOTE on Validation Set:")
    print(f"Total Validation Transactions: {len(y_val_smote)}")
    print(f"Fraud Cases: {sum(y_val_smote)}")
    print(f"Fraud Percentage: {y_val_smote.mean()*100:.2f}%")

    # Apply SMOTE to test set
    X_test_smote, y_test_smote = smote.fit_resample(X_test, y_test)
    print(f"\nAfter SMOTE on Test Set:")
    print(f"Total Test Transactions: {len(y_test_smote)}")
    print(f"Fraud Cases: {sum(y_test_smote)}")
    print(f"Fraud Percentage: {y_test_smote.mean()*100:.2f}%")

    # Train only on normal transactions
    X_train_normal = X_train[y_train == 0]

    # Initialize and train model
    unet_model = FraudDetectionUNet(hidden_layers=(64, 32, 16), learning_rate=0.0005)
    unet_model.create_unet_model(input_dim=X_train.shape[1])
    history = unet_model.fit(X_train_normal, X_val_smote, y_val_smote, epochs=50)

    # Final evaluation on SMOTE-balanced test set
    print("\nFinal Test Set Evaluation (with SMOTE):")
    unet_model.evaluate_model(X_test_smote, y_test_smote)

if __name__ == "__main__":
    main()