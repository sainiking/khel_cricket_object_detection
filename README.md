# Hybrid Cricket Detection Project

## Simple Overview
This project uses **YOLOv8** and **sklearn-style patterns** to detect cricket objects AND cricket ball in videos. Perfect for learning computer vision with hybrid detection!

## What It Detects
The hybrid model is trained to detect **cricket ball + 12 cricket-specific objects**:
- **Cricket Ball** (high priority detection)
- Bails, Batter, Batting Pads, Boundary Line
- Bowler, Fielder, Helmet, Non-Striker  
- Stumps, Stumps Mic, Umpire, Wicket Keeper

## Files Explained

### 1. `headless_frame_extractor.py` 
**Frame Extraction** - Prepare data for annotation
- Extracts frames from cricket videos for annotation
- Creates `extracted_frames_[video_name]/` folders
- Saves frames as `frame_0001_1.5s.jpg` format
- Use this to prepare your own dataset from cricket videos (include cricket ball)

### 2. `train_custom_cricket_model.py` 
**Hybrid Training Script** - Like training a sklearn model
- Loads cricket images and labels from Roboflow
- Uses `train_test_split()` from sklearn for data splitting
- Trains YOLOv8 to recognize cricket objects AND cricket ball
- Creates unified hybrid model for complete cricket analysis

### 3. `simple_cricket_detector.py`
**Hybrid Detection Script** - Like using a trained sklearn model
- Loads your trained hybrid model
- Processes cricket videos frame by frame
- Shows detection boxes around cricket objects AND ball
- Simple, clean interface with ball tracking

## How to Use

### Option A: Use Existing Dataset (Quick Start)
If you already have the `roboflow_dataset/` folder with ball annotations:

```bash
# Train hybrid model directly
python3 train_custom_cricket_model.py

# Run hybrid detection
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

#### Step 2: Annotate Frames (Include Cricket Ball)
1. Upload extracted frames to [Roboflow.com](https://roboflow.com)
2. Create project: "Hybrid Cricket Detection" 
3. Annotate cricket ball + 12 cricket objects in each frame
4. Export dataset in YOLOv8 format
5. Download and extract as `roboflow_dataset/`

#### Step 3: Train Hybrid Model
```bash
python3 train_custom_cricket_model.py
```

#### Step 4: Test Hybrid Detection
```bash
python3 simple_cricket_detector.py
```

## Complete Hybrid Data Pipeline
1. **Videos** → `headless_frame_extractor.py` → **Frames**
2. **Frames** → Roboflow annotation (ball + objects) → **Hybrid Labeled Dataset** 
3. **Hybrid Labeled Dataset** → `train_custom_cricket_model.py` → **Hybrid Trained Model**
4. **Hybrid Trained Model** → `simple_cricket_detector.py` → **Cricket Ball + Object Detection**

## Key Features
- **Hybrid Detection**: Cricket ball + cricket objects in one model
- **Simple Code**: Uses sklearn patterns you already know
- **Clear Comments**: Every function explained
- **Beginner Friendly**: Easy to understand and explain
- **GPU Optimized**: Works great with GTX 1660 Ti
- **Real-time**: Processes videos smoothly with ball tracking

## Project Structure
```
khel/
├── train_custom_cricket_model.py    # Hybrid Training (like sklearn fit)
├── simple_cricket_detector.py       # Hybrid Detection (like sklearn predict)  
├── roboflow_dataset/                # Your cricket images & labels (with ball)
├── cricket_runs/                    # Trained hybrid models saved here
└── your_videos.mp4                  # Cricket videos to test
```

## Why This Hybrid Approach?
- **Complete**: Detects both cricket ball and cricket objects
- **Unified**: One model handles everything (simpler than separate models)
- **Familiar**: Uses sklearn patterns for data splitting
- **Simple**: Clean, efficient implementation
- **Explainable**: Easy to present and explain
- **Reliable**: Focuses only on what works well
- **Educational**: Perfect for learning computer vision

## Technical Details
- **Framework**: YOLOv8 (ultralytics)
- **Data Split**: sklearn's train_test_split (80/20)
- **Model**: YOLOv8n (nano - fast hybrid training)
- **Classes**: Cricket ball + 12 cricket-specific objects
- **GPU**: Optimized for GTX 1660 Ti
- **Approach**: Unified hybrid detection (not multi-stage)

## Next Steps
1. Train your hybrid model with the cricket dataset (include ball annotations)
2. Test hybrid detection on cricket videos  
3. Present your working hybrid cricket detector!

Perfect for your first hybrid computer vision project!
