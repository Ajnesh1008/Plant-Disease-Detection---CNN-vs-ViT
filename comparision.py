import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle
from sklearn.preprocessing import label_binarize
from tqdm import tqdm  # For progress bar with ETA
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import os

# Re-import local model definitions
from cnn_model import CNNModel
from vit_model import ViTModel

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load test dataset
test_data = datasets.ImageFolder("data/plant_disease_dataset/val", transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
class_names = test_data.classes
n_classes = len(class_names)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNNModel().to(device)
vit_model = ViTModel().to(device)
cnn_model.load_state_dict(torch.load("models/cnn_model.pth", map_location=device))
vit_model.load_state_dict(torch.load("models/vit_model.pth", map_location=device))
cnn_model.eval()
vit_model.eval()

# Initialize containers
y_true, cnn_preds, vit_preds = [], [], []
cnn_scores, vit_scores = [], []

# Initialize tqdm for progress bar
total_batches = len(test_loader)
progress_bar = tqdm(test_loader, total=total_batches, desc="Evaluating", ncols=100, unit="batch")

# Evaluation loop with ETA
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

        # Update the progress bar (tqdm handles ETA automatically)
        progress_bar.set_postfix({'batch': progress_bar.n, 'total': total_batches})

# Convert to numpy arrays
y_true = np.array(y_true)
cnn_preds = np.array(cnn_preds)
vit_preds = np.array(vit_preds)
cnn_scores = np.array(cnn_scores)
vit_scores = np.array(vit_scores)

# Function to plot ROC curves for multiclass
def plot_roc(y_true, y_score, title):
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {class_names[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    plt.show()

# Plot ROC curves
plot_roc(y_true, cnn_scores, "CNN ROC Curve (Multiclass)")
plot_roc(y_true, vit_scores, "ViT ROC Curve (Multiclass)")







# Ensure output directory exists
os.makedirs("results", exist_ok=True)

def get_metrics(y_true, y_pred, average='weighted'):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average),
        "Recall": recall_score(y_true, y_pred, average=average),
        "F1-Score": f1_score(y_true, y_pred, average=average)
    }

def plot_conf_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_conf_matrix.png")
    plt.close()

def plot_metric_bars(df_scores):
    df_scores.plot(kind='bar', figsize=(10, 6), colormap='Set2')
    plt.title("CNN vs ViT - Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("results/metric_comparison_bar_chart.png")
    plt.close()

def plot_roc(y_true, y_score, class_names, model_name):
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(plt.cm.tab10.colors)
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title(f"{model_name} - ROC Curve (Multiclass)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    plt.savefig(f"results/{model_name}_roc_curve.png")
    plt.close()

def compare_models(y_true, cnn_pred, vit_pred, cnn_proba, vit_proba, class_names):
    # Time measurement
    start = time.time()
    cnn_metrics = get_metrics(y_true, cnn_pred)
    cnn_time = time.time() - start

    start = time.time()
    vit_metrics = get_metrics(y_true, vit_pred)
    vit_time = time.time() - start

    # Append runtime to metrics
    cnn_metrics["Inference Time (s)"] = round(cnn_time, 4)
    vit_metrics["Inference Time (s)"] = round(vit_time, 4)

    df_scores = pd.DataFrame([cnn_metrics, vit_metrics], index=["CNN", "ViT"])
    print("\n🔍 Model Comparison Metrics:")
    print(df_scores)

    # Export to CSV
    df_scores.to_csv("results/model_comparison_metrics.csv")

    # Generate plots
    plot_conf_matrix(y_true, cnn_pred, class_names, "CNN")
    plot_conf_matrix(y_true, vit_pred, class_names, "ViT")
    plot_metric_bars(df_scores)
    plot_roc(y_true, cnn_proba, class_names, "CNN")
    plot_roc(y_true, vit_proba, class_names, "ViT")

    # Write summary report
    with open("results/comparison_report.txt", "w") as f:
        f.write("Model Comparison Report\n")
        f.write("="*50 + "\n\n")
        f.write(df_scores.to_string() + "\n\n")
        f.write("Confusion matrices and ROC curves saved as images.\n")

    print("\n📁 Results exported to the 'results' folder.")

# ----------------------
# ⬇️ Example usage ⬇️
# compare_models(y_true, cnn_preds, vit_preds, cnn_probs, vit_probs, class_names)
# ----------------------

