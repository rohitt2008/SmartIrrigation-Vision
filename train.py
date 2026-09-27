import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import LightResNet50
from dataset_loader import get_data_loaders, TomatoLeafDataset, get_transforms, get_image_paths_and_labels

def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
    val_loss = val_loss / len(loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate macro precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    return val_loss, accuracy, precision, recall, f1, all_preds, all_labels

def plot_training_history(train_losses, val_losses, train_accs, val_accs, val_precs, val_recs, val_f1s, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', marker='o')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Metrics plot (Accuracy, Precision, Recall, F1)
    ax2.plot(epochs, train_accs, 'b--', label='Training Accuracy', marker='x')
    ax2.plot(epochs, val_accs, 'g-', label='Validation Accuracy', marker='o')
    ax2.plot(epochs, val_precs, 'm-', label='Val Precision', marker='s')
    ax2.plot(epochs, val_recs, 'c-', label='Val Recall', marker='^')
    ax2.plot(epochs, val_f1s, 'y-', label='Val F1', marker='d')
    ax2.set_title('Metrics over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_class_metrics(labels, preds, class_names, save_path):
    # Calculate per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    rects2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon')
    
    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Metrics per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(loc='lower right')
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train LightResNet50 Tomato Irrigation Status Model")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick training dry-run")
    args = parser.parse_args()
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    class_names = ["LI", "MI", "HI", "FI"]
    
    num_folds = 5 if not args.dry_run else 1
    fold_results = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'kappa': []}
    
    for fold in range(num_folds):
        print(f"\n{'='*40}")
        print(f"FOLD {fold + 1}/{num_folds}")
        print(f"{'='*40}")
        
        current_seed = 42 + fold
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        if args.dry_run:
            print("\n=== DRY RUN MODE ===")
            file_paths, labels = get_image_paths_and_labels(dataset_dir)
            if len(file_paths) == 0:
                print("No real images found for dry-run. Creating a mock loader...")
                class MockTomatoDataset(torch.utils.data.Dataset):
                    def __init__(self, size=20, transform=None):
                        self.size = size
                    def __len__(self): return self.size
                    def __getitem__(self, idx): return torch.randn(3, 224, 224), idx % 4
                        
                train_loader = torch.utils.data.DataLoader(MockTomatoDataset(100), batch_size=4, shuffle=True)
                val_loader = torch.utils.data.DataLoader(MockTomatoDataset(20), batch_size=4, shuffle=False)
                test_loader = torch.utils.data.DataLoader(MockTomatoDataset(20), batch_size=4, shuffle=False)
            else:
                subset_paths = []
                subset_labels = []
                for class_label in [0, 1, 2, 3]:
                    class_indices = [i for i, l in enumerate(labels) if l == class_label]
                    for idx in class_indices[:30]:
                        subset_paths.append(file_paths[idx])
                        subset_labels.append(labels[idx])
                
                train_transform, val_test_transform = get_transforms()
                
                from sklearn.model_selection import train_test_split
                p_train_val, p_test, l_train_val, l_test = train_test_split(
                    subset_paths, subset_labels, test_size=0.30, stratify=subset_labels, random_state=current_seed
                )
                p_train, p_val, l_train, l_val = train_test_split(
                    p_train_val, l_train_val, test_size=0.20, stratify=l_train_val, random_state=current_seed
                )
                
                train_dataset = TomatoLeafDataset(p_train, l_train, transform=train_transform)
                val_dataset = TomatoLeafDataset(p_val, l_val, transform=val_test_transform)
                test_dataset = TomatoLeafDataset(p_test, l_test, transform=val_test_transform)
                
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
        else:
            print(f"Loading dataset from {dataset_dir} with seed {current_seed}...")
            try:
                train_loader, val_loader, test_loader = get_data_loaders(
                    dataset_dir, batch_size=args.batch_size, random_state=current_seed
                )
            except Exception as e:
                print(f"[ERROR] Failed to load dataset: {e}")
                return

        model = LightResNet50(num_classes=4).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = args.patience
        
        history_train_loss = []
        history_val_loss = []
        history_train_acc = []
        history_val_acc = []
        history_val_prec = []
        history_val_rec = []
        history_val_f1 = []
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            loop = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch}/{args.epochs}")
            for inputs, targets in loop:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == targets).sum().item()
                loop.set_postfix(loss=loss.item())
                
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / len(train_loader.dataset)
            
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            
            history_train_loss.append(train_loss)
            history_val_loss.append(val_loss)
            history_train_acc.append(train_acc)
            history_val_acc.append(val_acc)
            history_val_prec.append(val_prec)
            history_val_rec.append(val_rec)
            history_val_f1.append(val_f1)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(base_dir, f"best_model_fold{fold+1}.pth"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
                    
        # Load best model for fold evaluation
        best_model_path = os.path.join(base_dir, f"best_model_fold{fold+1}.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            
        test_loss, test_acc, test_prec, test_rec, test_f1, preds, labels = evaluate(
            model, test_loader, criterion, device
        )
        
        kappa = cohen_kappa_score(labels, preds)
        cm = confusion_matrix(labels, preds)
        
        print(f"Fold {fold+1} Results: Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Kappa: {kappa:.4f}")
        fold_results['acc'].append(test_acc)
        fold_results['prec'].append(test_prec)
        fold_results['rec'].append(test_rec)
        fold_results['f1'].append(test_f1)
        fold_results['kappa'].append(kappa)
        
        # Save plots for the last fold
        if fold == num_folds - 1:
            plot_training_history(history_train_loss, history_val_loss, history_train_acc, history_val_acc, 
                                  history_val_prec, history_val_rec, history_val_f1,
                                  os.path.join(base_dir, "training_history.png"))
            plot_confusion_matrix(cm, class_names, os.path.join(base_dir, "confusion_matrix.png"))
            plot_class_metrics(labels, preds, class_names, os.path.join(base_dir, "class_metrics.png"))

    print("\n" + "="*40)
    print("FINAL 5-FOLD CROSS-VALIDATION RESULTS:")
    print("="*40)
    avg_acc = np.mean(fold_results['acc'])
    avg_prec = np.mean(fold_results['prec'])
    avg_rec = np.mean(fold_results['rec'])
    avg_f1 = np.mean(fold_results['f1'])
    avg_kappa = np.mean(fold_results['kappa'])
    
    print(f"Accuracy:      {avg_acc * 100:.2f}% (± {np.std(fold_results['acc'])*100:.2f}%)")
    print(f"Precision:     {avg_prec * 100:.2f}% (± {np.std(fold_results['prec'])*100:.2f}%)")
    print(f"Recall:        {avg_rec * 100:.2f}% (± {np.std(fold_results['rec'])*100:.2f}%)")
    print(f"F1-Score:      {avg_f1 * 100:.2f}% (± {np.std(fold_results['f1'])*100:.2f}%)")
    print(f"Kappa Score:   {avg_kappa:.4f} (± {np.std(fold_results['kappa']):.4f})")
    print("="*40)
    
    report_path = os.path.join(base_dir, "cv_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Tomato Irrigation Status Model - 5-Fold CV Report ===\n")
        f.write(f"Accuracy:      {avg_acc * 100:.2f}% ± {np.std(fold_results['acc'])*100:.2f}%\n")
        f.write(f"Precision (Macro): {avg_prec * 100:.2f}% ± {np.std(fold_results['prec'])*100:.2f}%\n")
        f.write(f"Recall (Macro):    {avg_rec * 100:.2f}% ± {np.std(fold_results['rec'])*100:.2f}%\n")
        f.write(f"F1-Score (Macro):  {avg_f1 * 100:.2f}% ± {np.std(fold_results['f1'])*100:.2f}%\n")
        f.write(f"Cohen's Kappa:     {avg_kappa:.4f} ± {np.std(fold_results['kappa']):.4f}\n")
    print(f"Saved text evaluation report to {report_path}")
    print("\nTo generate Grad-CAM visual heatmaps, please run: python3 gradcam_visualizer.py")

if __name__ == "__main__":
    main()
