import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import cycle
from tqdm import tqdm

from cnn_model import CNNModel
from vit_model import ViTModel

# Transform and Load Data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_data = datasets.ImageFolder("data/plant_disease_dataset/val", transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
class_names = test_data.classes
n_classes = len(class_names)

# Load Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNNModel().to(device)
vit_model = ViTModel().to(device)
cnn_model.load_state_dict(torch.load("models/cnn_model.pth", map_location=device))
vit_model.load_state_dict(torch.load("models/vit_model.pth", map_location=device))
cnn_model.eval()
vit_model.eval()

# Prediction Phase
y_true, cnn_preds, vit_preds = [], [], []
cnn_scores, vit_scores = [], []
progress_bar = tqdm(test_loader, total=len(test_loader), desc="Evaluating", ncols=100, unit="batch")

with torch.no_grad():
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        cnn_output = cnn_model(images)
        vit_output = vit_model(images)

        y_true.extend(labels.cpu().numpy())
        cnn_preds.extend(torch.argmax(cnn_output, dim=1).cpu().numpy())
        vit_preds.extend(torch.argmax(vit_output, dim=1).cpu().numpy())
        cnn_scores.extend(cnn_output.cpu().numpy())
        vit_scores.extend(vit_output.cpu().numpy())

# Convert predictions to NumPy arrays
y_true = np.array(y_true)
cnn_preds = np.array(cnn_preds)
vit_preds = np.array(vit_preds)
cnn_scores = np.array(cnn_scores)
vit_scores = np.array(vit_scores)

# === ROC Curve Function ===
def plot_roc(y_true, y_score, model_name):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_name} ROC Curve")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.show()

# === Metrics Reporting and Confusion Matrix ===

def generate_metrics_report(y_true, preds, model_name):
    report = classification_report(y_true, preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, preds)

    # Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Convert report to DataFrame
    df_report = pd.DataFrame(report).transpose()

    # Drop 'accuracy', 'macro avg', and 'weighted avg' for class-level plots
    df_plot = df_report.drop(index=["accuracy", "macro avg", "weighted avg"], errors='ignore')

    # Plot Precision, Recall, F1-Score per class
    metrics_to_plot = ["precision", "recall", "f1-score"]
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=df_plot.index, y=df_plot[metric], palette="Set2")
        plt.title(f"{model_name} - {metric.capitalize()} per Class")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Class")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    # Add model name for export tracking
    df_report['model'] = model_name
    return df_report


# === Generate Reports ===
cnn_report = generate_metrics_report(y_true, cnn_preds, "CNN")
vit_report = generate_metrics_report(y_true, vit_preds, "ViT")

# Plot ROC Curves
plot_roc(y_true, cnn_scores, "CNN")
plot_roc(y_true, vit_scores, "ViT")

# Export Combined Report
combined_report = pd.concat([cnn_report, vit_report])
combined_report.to_csv("model_comparison_report.csv", index=True)
print("✅ Exported metrics to model_comparison_report.csv")
