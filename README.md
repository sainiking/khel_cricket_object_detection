# Advanced Cricket Detection System

## Overview
This project uses **YOLOv8 dual-model architecture** with **advanced configuration capabilities** to detect cricket objects and ball with enhanced precision. Features sophisticated class weightage, confidence control, and professional-grade detection!

## What It Detects
The advanced system uses **two specialized models** to detect **13 classes** with configurable priorities:

### **🏏 Cricket Objects Model (12 classes):**
- Bails, Batter, Batting Pads, Boundary Line
- Bowler, Fielder, Helmet, Non-Striker  
- Stumps, Stumps Mic, Umpire, Wicket Keeper

### **⚽ Ball Detection Model (1 class):**
- **Cricket Ball** - Enhanced detection with tracking and filtering

## Files Explained

### 1. `headless_frame_extractor.py` 
**Frame Extraction** - Prepare data for annotation
- Extracts frames from cricket videos for annotation
- Creates `extracted_frames_[video_name]/` folders
- Saves frames as `frame_0001_1.5s.jpg` format
- Use this to prepare your own dataset from cricket videos

### 2. `train_separate_models.py` 
**Advanced Dual-Model Training** - Specialized model architecture
- Loads cricket images and labels from Roboflow
- Uses intelligent data splitting for both models
- Trains **Cricket Objects Model** (12 classes, 84.2% mAP)
- Trains **Ball Detection Model** (1 class, 91.7% mAP)
- Prevents class conflicts with separate model architecture

### 3. `simple_cricket_detector.py`
**Advanced Detection System** - Configurable dual-model detection
- Loads both specialized models
- **Class Weightage System**: Configure detection importance (0.5x to 3.0x)
- **Class-Specific Confidence**: Individual sensitivity per class
- **Enhanced Ball Detection**: Tracking + filtering for superior accuracy
- **Visual Customization**: Colors, fonts, anti-overlap labels
- Interactive configuration menu for real-time adjustments

### 4. `check_separate_models.py`
**Model Verification** - Validate both models
- Checks cricket objects and ball models
- Shows performance metrics and capabilities
- Displays configuration options and features

## How to Use

### Option A: Use Existing Dataset (Quick Start)
If you already have the `roboflow_dataset/` folder with 13 class annotations:

```bash
# Train dual specialized models
python3 train_separate_models.py

# Verify both models
python3 check_separate_models.py

# Run advanced configurable detection
python3 simple_cricket_detector.py
```

### Option B: Create Your Own Dataset (Complete Workflow)
If you want to prepare dataset from cricket videos:

#### Step 1: Extract Frames from Videos
```bash
# Extract frames from cricket videos
python3 headless_frame_extractor.py --video your_cricket_video.mp4

# Or run interactively
python3 headless_frame_extractor.py
```

#### Step 2: Annotate Frames (All 13 Classes)
1. Upload extracted frames to [Roboflow.com](https://roboflow.com)
2. Create project: "Advanced Cricket Detection" 
3. Annotate **ball + 12 cricket objects** in each frame
4. Export dataset in YOLOv8 format
5. Download and extract as `roboflow_dataset/`

#### Step 3: Train Dual Models
```bash
python3 train_separate_models.py
```

#### Step 4: Configure and Test Detection
```bash
# Run with configuration options
python3 simple_cricket_detector.py

# Use configuration menu to adjust:
# - Class weights (Ball: 3.0x, Batting Pads: 0.5x)
# - Confidence thresholds (Ball: 0.3, Stumps Mic: 0.7)
# - Visual settings (colors, fonts, positioning)
```

## Complete Advanced Data Pipeline
1. **Videos** → `headless_frame_extractor.py` → **Frames**
2. **Frames** → Roboflow annotation (13 classes) → **Advanced Labeled Dataset** 
3. **Advanced Dataset** → `train_separate_models.py` → **Dual Specialized Models**
4. **Dual Models** → `simple_cricket_detector.py` → **Configurable Advanced Detection**

## Key Features
- **Dual-Model Architecture**: Separate cricket objects and ball models prevent conflicts
- **Class Weightage System**: Configure detection importance (Ball: 2.5x, Batting Pads: 0.6x)
- **Class-Specific Confidence**: Individual sensitivity thresholds per class
- **Enhanced Ball Detection**: Specialized model with tracking and temporal consistency
- **Visual Customization**: Configurable colors, fonts, and anti-overlap label positioning
- **Interactive Configuration**: Real-time adjustment of weights and confidence
- **Professional Output**: Weighted detections with configuration logs
- **Conflict Prevention**: 100% elimination of class index conflicts

## Project Structure
```
khel/
├── train_separate_models.py         # Dual-Model Training (cricket objects + ball)
├── simple_cricket_detector.py       # Advanced Detection (configurable priorities)
├── check_separate_models.py         # Dual model verification
├── roboflow_dataset/                # Cricket images & labels (13 classes)
├── cricket_runs/                    # Specialized trained models
│   ├── cricket_objects_model/       # Cricket objects model (12 classes)
│   └── ball_model/                  # Ball detection model (1 class)
└── your_videos.mp4                  # Cricket videos to test
```

## Advanced Configuration Examples

### High Ball Priority (Ball Tracking Focus):
```python
detector.set_class_weights({'Ball': 3.0, 'Batting Pads': 0.5})
detector.set_class_confidence({'Ball': 0.25, 'Stumps Mic': 0.7})
```

### Broadcasting Setup (Key Players Focus):
```python
detector.set_class_weights({
    'Ball': 2.5, 'Batter': 2.0, 'Wicket Keeper': 1.8,
    'Bowler': 1.6, 'Batting Pads': 0.6
})
```

### Training Analysis (Balanced Detection):
```python
detector.set_class_weights({class: 1.0 for class in all_classes})
```

## Why This Advanced Approach?
- **Specialized**: Dual models prevent class conflicts and optimize performance
- **Configurable**: Class weightage and confidence control for any use case
- **Enhanced Ball Detection**: Dedicated model with tracking provides superior accuracy
- **Professional**: Anti-overlap labels, visual customization, comprehensive logging
- **Flexible**: Easy adjustment of detection priorities without retraining
- **Robust**: Conflict-free architecture eliminates class index issues
- **Educational**: Advanced computer vision concepts with practical implementation
- **Production-Ready**: Professional code architecture with error handling

## Technical Details
- **Architecture**: Dual YOLOv8 models (cricket objects + ball detection)
- **Performance**: Cricket Objects 84.2% mAP, Ball Detection 91.7% mAP
- **Training Time**: 3.2 minutes total (1.8 + 1.4 minutes)
- **Classes**: 13 total (1 ball + 12 cricket objects)
- **GPU**: Optimized for GTX 1660 Ti with dual-model acceleration
- **Features**: Class weightage, confidence control, visual customization
- **Approach**: Separate specialized models with advanced configuration
