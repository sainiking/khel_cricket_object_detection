#!/usr/bin/env python3
"""
Check Separate Cricket Models
Verifies that both cricket models are available and shows their capabilities
"""

import sys
from pathlib import Path

def check_cricket_objects_model():
    """Check for cricket objects model"""
    
    model_paths = [
        "cricket_runs/cricket_objects_model/weights/best.pt",
        "cricket_runs/cricket_model_v1/weights/best.pt",
        "runs/detect/cricket_objects_model/weights/best.pt",
        "runs/detect/cricket_model_v1/weights/best.pt"
    ]
    
    for path in model_paths:
        if Path(path).exists():
            return path, True
    
    return None, False

def check_ball_detection_model():
    """Check for ball detection model"""
    
    # Check for custom ball model first
    custom_ball_paths = [
        "cricket_runs/cricket_ball_model/weights/best.pt",
        "runs/detect/cricket_ball_model/weights/best.pt"
    ]
    
    for path in custom_ball_paths:
        if Path(path).exists():
            return path, True, "custom"
    
    # General YOLO model is always available (downloads automatically)
    return "yolov8n.pt", True, "general"

def show_model_info():
    """Display information about both models"""
    
    try:
        from ultralytics import YOLO
        
        print("Cricket Objects Model:")
        print("-" * 30)
        
        objects_path, objects_found = check_cricket_objects_model()
        
        if objects_found:
            print(f"✓ Found: {objects_path}")
            
            try:
                model = YOLO(objects_path)
                if hasattr(model.model, 'names'):
                    class_names = list(model.model.names.values())
                    print(f"  Classes: {len(class_names)}")
                    for i, name in enumerate(class_names):
                        print(f"     {i:2d}. {name}")
            except Exception as e:
                print(f"  Error loading model: {e}")
        else:
            print("✗ Not found")
            print("  Train with: python3 train_separate_models.py")
        
        print("\nBall Detection Model:")
        print("-" * 30)
        
        ball_path, ball_found, ball_type = check_ball_detection_model()
        
        if ball_found:
            if ball_type == "custom":
                print(f"✓ Custom ball model: {ball_path}")
            else:
                print(f"✓ General YOLO model: {ball_path}")
                print("  Detects sports balls (class 32)")
        else:
            print("✗ No ball detection available")
        
        return objects_found and ball_found
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Main function to check separate models"""
    
    print("Separate Cricket Models Check")
    print("=" * 50)
    print("Checking for TWO separate models:")
    print("1. Cricket Objects Model (custom trained)")
    print("2. Ball Detection Model (custom or general)")
    print("-" * 50)
    
    success = show_model_info()
    
    print("\n" + "=" * 50)
    
    if success:
        print("✓ Models ready for detection!")
        print("\nNext steps:")
        print("   1. Run detection: python3 simple_cricket_detector.py")
        print("   2. Models will work together without class conflicts")
    else:
        print("⚠ Some models missing")
        print("\nTo fix:")
        print("   1. Train cricket objects: python3 train_separate_models.py")
        print("   2. Ball detection will use general YOLO model")
    
    print("\nAdvantages of separate models:")
    print("   • No class index conflicts")
    print("   • Better accuracy for each task")
    print("   • Can optimize each model independently")

if __name__ == "__main__":
    main()
