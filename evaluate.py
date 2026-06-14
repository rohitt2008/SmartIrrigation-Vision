import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import LightResNet50
from dataset_loader import get_data_loaders
from train import evaluate, plot_confusion_matrix, plot_class_metrics

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    best_model_path = os.path.join(base_dir, "best_model.pth")
    class_names = ["LI", "MI", "HI", "FI"]
    
    if not os.path.exists(best_model_path):
        print("No trained model (best_model.pth) found. Please run train.py first.")
        return
        
    print("Loading test dataset...")
    # We only need the test loader, but get_data_loaders returns all 3.
    _, _, test_loader = get_data_loaders(dataset_dir, batch_size=64, random_state=42)
    
    print("Loading best_model.pth...")
    model = LightResNet50(num_classes=4).to(device)
    model.load_state_dict(torch.load(best_model_path))
    criterion = nn.CrossEntropyLoss()
    
    print("Evaluating model and generating plots...")
    _, test_acc, test_prec, test_rec, test_f1, preds, labels = evaluate(
        model, test_loader, criterion, device
    )
    
    cm = confusion_matrix(labels, preds)
    
    # Generate Plots!
    plot_confusion_matrix(cm, class_names, os.path.join(base_dir, "confusion_matrix.png"))
    plot_class_metrics(labels, preds, class_names, os.path.join(base_dir, "class_metrics.png"))
    
    print(f"\nSuccessfully generated plots!")
    print(f"Accuracy: {test_acc * 100:.2f}%")
    print(f"Check your folder for: ")
    print(f" -> confusion_matrix.png")
    print(f" -> class_metrics.png")

if __name__ == "__main__":
    main()
