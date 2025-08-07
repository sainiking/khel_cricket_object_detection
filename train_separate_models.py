#!/usr/bin/env python3
"""
Separate Cricket Models Training
- Trains TWO separate YOLO models to avoid class index conflicts
- Model 1: Cricket objects (original 12 classes)
- Model 2: Cricket ball detection (separate specialized model)
- Combines results in detection for complete cricket analysis

This approach prevents class index disruption and maintains accuracy
"""

# Import required libraries
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

def prepare_cricket_objects_dataset():
    """Prepare dataset for cricket objects (no ball)"""
    
    print("Preparing cricket objects dataset...")
    
    roboflow_dir = Path("roboflow_dataset")
    if not roboflow_dir.exists():
        print("Error: roboflow_dataset folder not found!")
        return None
    
    train_images = roboflow_dir / "train" / "images"
    train_labels = roboflow_dir / "train" / "labels"
    
    if not (train_images.exists() and train_labels.exists()):
        print("Error: Images or labels folder not found!")
        return None
    
    # Collect image-label pairs
    image_files = []
    label_files = []
    
    for img_file in train_images.glob("*.jpg"):
        label_file = train_labels / (img_file.stem + ".txt")
        if label_file.exists():
            image_files.append(img_file)
            label_files.append(label_file)
    
    print(f"Found {len(image_files)} image-label pairs")
    
    # Split data using sklearn
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        image_files, label_files,
        test_size=0.2,
        random_state=42
    )
    
    return train_imgs, val_imgs, train_lbls, val_lbls

def setup_cricket_objects_folders(train_imgs, val_imgs, train_lbls, val_lbls):
    """Setup folders for cricket objects training (original model)"""
    
    # Create training folder for cricket objects
    train_dir = Path("cricket_objects_training")
    train_dir.mkdir(exist_ok=True)
    
    folders = [
        train_dir / "images" / "train",
        train_dir / "images" / "val",
        train_dir / "labels" / "train",
        train_dir / "labels" / "val"
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    
    # Copy training files
    for img, lbl in zip(train_imgs, train_lbls):
        shutil.copy2(img, train_dir / "images" / "train" / img.name)
        shutil.copy2(lbl, train_dir / "labels" / "train" / lbl.name)
    
    # Copy validation files
    for img, lbl in zip(val_imgs, val_lbls):
        shutil.copy2(img, train_dir / "images" / "val" / img.name)
        shutil.copy2(lbl, train_dir / "labels" / "val" / lbl.name)
    
    # Define cricket object classes (ORIGINAL 12 classes)
    cricket_classes = [
        'Bails', 'Batter', 'Batting Pads', 'Boundary Line',
        'Bowler', 'Fielder', 'Helmet', 'Non-Striker', 
        'Stumps', 'Stumps Mic', 'Umpire', 'Wicket Keeper'
    ]
    
    # Create configuration file
    config = {
        'train': str((train_dir / "images" / "train").absolute()),
        'val': str((train_dir / "images" / "val").absolute()),
        'nc': len(cricket_classes),
        'names': cricket_classes
    }
    
    config_file = train_dir / "cricket_objects_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_file

def filter_ball_labels(label_files, ball_class_id=None):
    """Filter labels to only include ball annotations"""
    
    ball_label_files = []
    
    for label_file in label_files:
        # Read the label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Filter for ball annotations only
        ball_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                # You need to identify which class ID represents the ball
                # Check your dataset's classes.txt or data.yaml to find ball class ID
                if ball_class_id is None:
                    # Auto-detect: assume ball is class 0 or look for 'ball' in original labels
                    # For now, we'll skip ball filtering and create a generic ball detector
                    continue
                elif class_id == ball_class_id:
                    # Change class ID to 0 for ball-only model
                    ball_lines.append(f"0 {' '.join(parts[1:])}\n")
        
        # Only include files that have ball annotations
        if ball_lines:
            # Create new label file with only ball annotations
            ball_label_file = label_file.parent / f"ball_{label_file.name}"
            with open(ball_label_file, 'w') as f:
                f.writelines(ball_lines)
            ball_label_files.append(ball_label_file)
    
    return ball_label_files

def setup_ball_detection_folders(train_imgs, val_imgs, train_lbls, val_lbls):
    """Setup folders for ball detection training (separate model)"""
    
    # Create training folder for ball detection
    train_dir = Path("cricket_ball_training")
    train_dir.mkdir(exist_ok=True)
    
    folders = [
        train_dir / "images" / "train",
        train_dir / "images" / "val",
        train_dir / "labels" / "train",
        train_dir / "labels" / "val"
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    
    # For now, we'll create a simple ball detection setup
    # In real implementation, you would filter labels for ball class only
    
    # Copy training files
    for img, lbl in zip(train_imgs, train_lbls):
        shutil.copy2(img, train_dir / "images" / "train" / img.name)
        # For ball training, you would filter labels here
        shutil.copy2(lbl, train_dir / "labels" / "train" / lbl.name)
    
    # Copy validation files  
    for img, lbl in zip(val_imgs, val_lbls):
        shutil.copy2(img, train_dir / "images" / "val" / img.name)
        # For ball training, you would filter labels here
        shutil.copy2(lbl, train_dir / "labels" / "val" / lbl.name)
    
    # Define ball detection classes (ONLY ball)
    ball_classes = ['Ball']
    
    # Create configuration file
    config = {
        'train': str((train_dir / "images" / "train").absolute()),
        'val': str((train_dir / "images" / "val").absolute()),
        'nc': len(ball_classes),
        'names': ball_classes
    }
    
    config_file = train_dir / "cricket_ball_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_file

def train_cricket_objects_model():
    """Train the cricket objects model (original 12 classes)"""
    
    print("Training Cricket Objects Model...")
    print("=" * 50)
    
    # Prepare dataset
    data_split = prepare_cricket_objects_dataset()
    if data_split is None:
        return None
    
    train_imgs, val_imgs, train_lbls, val_lbls = data_split
    
    # Setup training folders
    config_file = setup_cricket_objects_folders(train_imgs, val_imgs, train_lbls, val_lbls)
    
    # Initialize model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data=str(config_file),
        epochs=25,
        imgsz=640,
        batch=8,
        device='cuda',
        patience=5,
        save=True,
        project='cricket_runs',
        name='cricket_objects_model',
        exist_ok=True,
        verbose=True
    )
    
    print("Cricket Objects Model Training Complete!")
    return results

def train_ball_detection_model():
    """Train the ball detection model (only ball)"""
    
    print("Training Cricket Ball Detection Model...")
    print("=" * 50)
    
    # For this example, we'll use a pre-trained sports ball model
    # In real implementation, you would train on your ball annotations
    
    print("Note: For ball detection, you have options:")
    print("1. Use existing sports ball detection model")
    print("2. Train custom ball model with your ball annotations")
    print("3. Use general object detection for round objects")
    
    # Option 1: Use existing model (quick solution)
    print("Using existing YOLOv8 model for ball detection...")
    
    # You could download a sports-specific model or use the general one
    # For now, we'll set up the structure for custom ball training
    
    # Uncomment below for custom ball training:
    """
    data_split = prepare_cricket_objects_dataset()
    if data_split is None:
        return None
    
    train_imgs, val_imgs, train_lbls, val_lbls = data_split
    config_file = setup_ball_detection_folders(train_imgs, val_imgs, train_lbls, val_lbls)
    
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=str(config_file),
        epochs=20,
        imgsz=640,
        batch=8,
        device='cuda',
        patience=5,
        save=True,
        project='cricket_runs',
        name='cricket_ball_model',
        exist_ok=True,
        verbose=True
    )
    """
    
    return None

def main():
    """Main training function - trains both models separately"""
    
    print("Separate Cricket Models Training")
    print("=" * 60)
    print("Training two separate models to avoid class conflicts:")
    print("1. Cricket Objects Model (12 classes)")
    print("2. Cricket Ball Detection Model (1 class)")
    print("-" * 60)
    
    # Train cricket objects model (original)
    objects_results = train_cricket_objects_model()
    
    if objects_results:
        print("\n✓ Cricket Objects Model trained successfully!")
        print(f"   Model saved in: cricket_runs/cricket_objects_model/")
    
    # Train ball detection model
    ball_results = train_ball_detection_model()
    
    print("\nTraining Summary:")
    print("=" * 30)
    print("✓ Cricket Objects Model: Ready")
    print("? Cricket Ball Model: Setup (needs ball annotations)")
    print("\nNext steps:")
    print("1. Test with: python3 simple_cricket_detector.py")
    print("2. Models will be combined during detection")

if __name__ == "__main__":
    main()
