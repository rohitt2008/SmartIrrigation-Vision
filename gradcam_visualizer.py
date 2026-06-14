import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import LightResNet50
from dataset_loader import get_image_paths_and_labels, get_transforms

def generate_gradcam_reports():
    print("Initializing Grad-CAM visualization report generator...")
    
    # Determine device
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
        print("ERROR: best_model.pth not found. Please train the model first.")
        return
        
    # Load Model
    print("Loading model architecture and weights...")
    model = LightResNet50(num_classes=4).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    # Target the last convolutional block (Block-E)
    # This is standard practice for ResNets to get the highest level feature maps
    target_layers = [model.block_e]
    
    # Initialize CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Get all images
    file_paths, labels = get_image_paths_and_labels(dataset_dir)
    
    _, val_test_transform = get_transforms()
    
    # Randomly select one image per class
    np.random.seed(42) # For reproducibility if needed
    selected_images = []
    
    for i in range(4):
        class_indices = [idx for idx, label in enumerate(labels) if label == i]
        if not class_indices:
            print(f"Warning: No images found for class {class_names[i]}")
            continue
        random_idx = np.random.choice(class_indices)
        selected_images.append((file_paths[random_idx], i))
        
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    fig.suptitle("Grad-CAM Visualization: AI Focus Regions by Irrigation Level", fontsize=16)
    
    print("Generating heatmaps...")
    for idx, (img_path, true_label) in enumerate(selected_images):
        print(f"Processing {class_names[true_label]} sample...")
        
        # Original Image (PIL)
        original_img = Image.open(img_path).convert('RGB')
        original_img = original_img.resize((224, 224))
        
        # Normalize original image to [0, 1] for CAM overlay
        rgb_img = np.float32(original_img) / 255.0
        
        # Prepare input tensor
        input_tensor = val_test_transform(original_img).unsqueeze(0).to(device)
        
        # Run prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, pred_idx = torch.max(output, 1)
            pred_class = pred_idx.item()
            
        # Target for CAM
        targets = [ClassifierOutputTarget(true_label)]
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay heatmap on original image
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Plotting
        axes[idx, 0].imshow(original_img)
        axes[idx, 0].set_title(f"Original: True {class_names[true_label]}, Pred {class_names[pred_class]}")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(cam_image)
        axes[idx, 1].set_title(f"Grad-CAM Heatmap (Focus)")
        axes[idx, 1].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    output_path = os.path.join(base_dir, "gradcam_results.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"\nSUCCESS! Grad-CAM visualization saved to {output_path}")
    print("This image highlights exactly which parts of the leaf the AI is using to determine the irrigation status.")

if __name__ == "__main__":
    generate_gradcam_reports()
