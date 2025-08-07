# Advanced Cricket Dataset Preparation Workflow

## Complete Data Pipeline for Dual-Model Architecture

This document explains how the cricket detection dataset was prepared, from raw videos to specialized trained models with enhanced ball detection.

## 🔄 Step-by-Step Workflow

### Step 1: Collect Cricket Videos
- Gather cricket match videos (MP4, AVI, MOV formats)
- Place videos in project folder
- Examples: `cricket_match_1.mp4`, `highlights.avi`

### Step 2: Extract Frames for Annotation
Use the frame extractor to get still images from videos:

```bash
# Extract frames from a specific video
python3 headless_frame_extractor.py --video cricket_match.mp4

# Extract frames interactively
python3 headless_frame_extractor.py

# Headless mode (automatic)
python3 headless_frame_extractor.py --headless --interval 30 --max-frames 100
```

**What this does:**
- Extracts 1 frame every 30 frames (1 frame per second at 30fps)
- Saves up to 100 frames per video
- Creates folder: `extracted_frames_[video_name]/`
- Frames named: `frame_0001_1.5s.jpg`, `frame_0002_2.0s.jpg`, etc.

### Step 3: Upload to Roboflow
1. Go to [Roboflow.com](https://roboflow.com)
2. Create new project: "Cricket Object Detection"
3. Upload extracted frames from `extracted_frames_*/`
4. Review and select best quality frames

### Step 4: Manual Annotation
Annotate cricket objects in each frame:

**13 Cricket Classes (Ball + Objects):**
1. **Ball** - Cricket ball (high priority for detection)
2. **Bails** - Small wooden pieces on stumps
3. **Batter** - Player with bat
4. **Batting Pads** - Leg protection gear
5. **Boundary Line** - Field boundary marking
6. **Bowler** - Player throwing ball
7. **Fielder** - Other players in field
8. **Helmet** - Head protection
9. **Non-Striker** - Batsman at other end
10. **Stumps** - Three wooden posts
11. **Stumps Mic** - Audio equipment on stumps
12. **Umpire** - Match referee
13. **Wicket Keeper** - Player behind stumps

**Annotation Tips:**
- Draw tight bounding boxes around objects
- **Ball annotation is critical** - mark even small/partially visible balls
- Be consistent with class names
- Skip blurry or unclear frames
- Aim for 15-20 annotations per class minimum
- **Special attention to ball visibility** - include frames with ball in motion

### Step 5: Export from Roboflow
1. Go to "Export" in your Roboflow project
2. Choose format: **YOLOv8**
3. Apply augmentations (optional):
   - Rotation: ±15 degrees
   - Brightness: ±25%
   - Blur: up to 1px
4. Download ZIP file

### Step 6: Setup Dataset
```bash
# Extract Roboflow zip file
unzip "Cricket Object Detection.v1i.yolov8.zip"

# Rename to expected folder name
mv "Cricket Object Detection-1" roboflow_dataset

# Verify structure
ls roboflow_dataset/
# Should see: train/ valid/ test/ data.yaml README.roboflow.txt
```

### Step 7: Train Separate Models
```bash
# Activate environment
source activate_cricket_env.sh

# Train specialized models (prevents class conflicts)
python3 train_separate_models.py

# This creates two models:
# 1. Cricket objects model (12 classes)
# 2. Ball detection model (1 class with enhanced detection)
```

### Step 8: Verify Models
```bash
# Check both models are trained correctly
python3 check_separate_models.py

# Displays:
# - Cricket objects model capabilities
# - Ball detection model performance  
# - Combined system overview
```

### Step 9: Advanced Detection
```bash
# Run dual-model detection with enhanced features
python3 simple_cricket_detector.py

# Features:
# - Separate models prevent class conflicts
# - Enhanced ball detection with tracking
# - Configurable class weightage
# - Class-specific confidence thresholds
# - Visual customization options
```

## Dataset Statistics

**Your Current Dataset:**
- **Total Images**: 216 annotated frames
- **Classes**: 13 total (1 ball + 12 cricket objects)
- **Split**: 80% training, 20% validation (sklearn style)
- **Source**: Cricket match videos → Frame extraction → Manual annotation
- **Architecture**: Dual-model system (cricket objects + ball detection)
- **Ball Focus**: Enhanced ball detection with specialized model

## Tools Used

### Frame Extraction
- **Tool**: `headless_frame_extractor.py`
- **Purpose**: Extract frames from cricket videos
- **Output**: Individual JPG images for annotation

### Annotation Platform
- **Tool**: Roboflow.com
- **Purpose**: Manual bounding box annotation
- **Output**: Labeled dataset in YOLO format

### Training Pipeline
- **Tool**: `train_separate_models.py`
- **Purpose**: Train two specialized YOLOv8 models
- **Output**: Cricket objects model + Ball detection model
- **Benefits**: Prevents class conflicts, enhanced ball detection

### Model Verification
- **Tool**: `check_separate_models.py`
- **Purpose**: Verify both models and show capabilities
- **Output**: Model status, performance metrics, feature overview

## File Structure

```
khel/
├── headless_frame_extractor.py     #  Extract frames from videos
├── train_separate_models.py        #  Train dual models (cricket + ball)
├── simple_cricket_detector.py      #  Run advanced detection
├── check_separate_models.py        #  Verify both models
├── roboflow_dataset/               #  Annotated dataset
│   ├── train/images/               #  Training images
│   ├── train/labels/               #  Training annotations
│   └── data.yaml                   #  Class configuration
├── extracted_frames_*/             #  Extracted frames (for annotation)
└── cricket_runs/                   #  Trained models
    ├── cricket_objects_model/      #  Cricket objects model (12 classes)
    └── ball_model/                 #  Ball detection model (1 class)
```

## Quality Tips

### Good Frames for Annotation:
-  Clear, well-lit images
-  Multiple cricket objects visible
-  **Cricket ball clearly visible** (even if small)
-  Different camera angles
-  Various game situations
-  Sharp focus
-  **Ball in different positions** (in air, in hand, on ground)

### Avoid:
-  Blurry or motion-blurred frames
-  Very dark or overexposed images
-  Frames with heavy occlusion
-  Duplicate or very similar frames
-  **Frames where ball is completely hidden**

##  Workflow Benefits

### Advanced Architecture Benefits:
1. **Dual Models**: Separate cricket objects and ball models prevent conflicts
2. **Enhanced Ball Detection**: Specialized model with tracking and filtering
3. **Configurable Detection**: Class weightage and confidence control
4. **Visual Customization**: Colors, fonts, and label positioning
5. **Production Ready**: Robust error handling and user-friendly interface

### Reproducible Process:
1. **Documented**: Clear steps from video → dual models
2. **Scalable**: Easy to add more videos/frames
3. **Quality Control**: Manual annotation ensures accuracy
4. **Professional**: Industry-standard annotation platform
5. **Advanced Features**: Class-specific tuning and visual control

### sklearn-style Training:
- Familiar data splitting patterns
- Clear train/validation separation
- Easy to understand and explain
- Separate model training for specialization

## Pro Tips

1. **Start Small**: Extract 50-100 frames initially
2. **Quality > Quantity**: Better to have fewer high-quality annotations
3. **Diverse Data**: Use frames from different matches/conditions
4. **Consistent Labels**: Use exact same class names throughout
5. **Regular Validation**: Test models frequently during development
6. **Ball Annotation Priority**: Ensure ball is annotated in various situations
7. **Class Balance**: Try to have similar numbers of each class
8. **Model Testing**: Use `check_separate_models.py` to verify training
9. **Configuration Tuning**: Adjust class weights and confidence as needed
10. **Visual Validation**: Use detection output to verify annotation quality

## Advanced Configuration Tips

### Class Weightage Recommendations:
- **High Priority**: Ball (2.5x), Batter (1.8x), Wicket Keeper (1.6x)
- **Medium Priority**: Bowler (1.4x), Fielder (1.2x), Stumps (1.0x)
- **Low Priority**: Batting Pads (0.6x), Stumps Mic (0.5x)

### Confidence Threshold Recommendations:
- **Sensitive**: Ball (0.3), Fielder (0.4), Batter (0.4)
- **Standard**: Bowler (0.4), Wicket Keeper (0.4), Stumps (0.5)
- **Less Sensitive**: Batting Pads (0.6), Stumps Mic (0.7)

---

**This advanced workflow ensures a professional, configurable process from raw cricket videos to specialized dual-model detection system with enhanced ball detection and user control!**
