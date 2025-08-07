#!/usr/bin/env python3
"""
Separate Cricket Detection Script
Uses TWO separate YOLO models to avoid class conflicts:
1. Cricket Objects Model (12 classes) - custom trained
2. Ball Detection Model (1 class) - general YOLO or custom

Similar to using multiple trained sklearn models:
1. Load both trained models
2. Process video frames with both models
3. Combine detection results without class conflicts

Detects:
- Cricket Objects: Bails, Batter, Batting Pads, Boundary Line,
  Bowler, Fielder, Helmet, Non-Striker, Stumps, Stumps Mic, Umpire, Wicket Keeper
- Cricket Ball: Using separate ball detection model
"""

import cv2 # OpenCV for video processing
from ultralytics import YOLO # YOLOv8 model for object detection
from pathlib import Path # For file and path handling
import glob # For file searching

class SeparateCricketDetector:
    """
    Separate cricket detector class
    Uses TWO separate YOLO models to avoid class conflicts:
    1. Cricket Objects Model (12 classes) 
    2. Ball Detection Model (general sports ball detection)
    """
    
    def __init__(self, objects_model_path=None, ball_model_path=None):
        """Initialize detector with two separate models"""
        
        if objects_model_path is None:
            objects_model_path = self.find_objects_model()
        
        print(f" Loading cricket objects model...")
        print(f" Objects Model: {objects_model_path}")
        
        # Load cricket objects model (original 12 classes)
        self.objects_model = YOLO(objects_model_path)
        
        # Load ball detection model (use general YOLO for sports ball)
        print(f" Loading ball detection model...")
        if ball_model_path is None:
            ball_model_path = self.find_ball_model()
        self.ball_model = YOLO(ball_model_path)
        
        # Define cricket object classes (original 12 classes)
        self.cricket_classes = [
            'Bails', 'Batter', 'Batting Pads', 'Boundary Line',
            'Bowler', 'Fielder', 'Helmet', 'Non-Striker', 
            'Stumps', 'Stumps Mic', 'Umpire', 'Wicket Keeper'
        ]
        
        # Colors for cricket objects (easily customizable unique colors)
        self.cricket_colors = {
            'Bails': (255, 165, 0),        # Orange
            'Batter': (255, 20, 147),      # Deep Pink
            'Batting Pads': (139, 69, 19), # Saddle Brown
            'Boundary Line': (0, 255, 0),  # Lime Green
            'Bowler': (255, 0, 255),       # Magenta
            'Fielder': (0, 255, 255),      # Cyan
            'Helmet': (128, 0, 255),       # Purple
            'Non-Striker': (255, 255, 0),  # Yellow
            'Stumps': (0, 0, 255),         # Blue
            'Stumps Mic': (255, 128, 0),   # Orange Red
            'Umpire': (0, 0, 139),         # Dark Blue
            'Wicket Keeper': (255, 0, 0)   # Red
        }
        
        # Font configuration for better text display
        self.font_config = {
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'scale': 0.7,           # Font size
            'thickness': 2,         # Font thickness
            'line_spacing': 25,     # Space between overlapping labels
            'background_alpha': 0.8 # Background transparency
        }
        
        # Ball detection color (bright green for high visibility)
        self.ball_color = (0, 255, 0)  # Bright Green - Ball
        
        # Ball tracking variables for consistency
        self.last_ball_position = None
        self.ball_tracking_frames = 0
        self.max_tracking_distance = 100  # Maximum distance ball can move between frames
        
        # Ball detection configuration (CONFIGURABLE)
        self.ball_detection_mode = "tracked"  # Options: "standard", "tracked"
        self.ball_weightage = 1.5             # Ball importance (1.0 = normal, >1.0 = more important)
        self.ball_confidence_threshold = 0.25  # Lower = more sensitive detection
        self.ball_confidence_boost = 1.2      # Multiply confidence by this factor
        
        # Class weightage system (1.0 = normal, <1.0 = less weightage, >1.0 = more weightage)
        self.class_weights = {
            'Bails': 1.0,
            'Batter': 1.5,           # Higher weightage for key player
            'Batting Pads': 0.5,     # REDUCED WEIGHTAGE - less important
            'Boundary Line': 0.8,
            'Bowler': 1.2,           # Higher weightage for key player
            'Fielder': 0.6,
            'Helmet': 0.7,
            'Non-Striker': 0.90,
            'Stumps': 1.1,           # Important cricket element
            'Stumps Mic': 0.6,
            'Umpire': 1.0,         # Lower weightage for less important
            'Wicket Keeper': 0.9     # Important cricket element
        }
        
        # Class-specific confidence thresholds (0.1-0.9, lower = more sensitive)
        self.class_confidence_thresholds = {
            'Bails': 0.3,           # Standard threshold
            'Batter': 0.70,         # Lower threshold for key player (more sensitive)
            'Batting Pads': 0.5,    # Higher threshold (less sensitive)
            'Boundary Line': 0.35,
            'Bowler': 0.45,         # Lower threshold for key player (more sensitive)
            'Fielder': 0.5,         # Slightly higher threshold
            'Helmet': 0.64,
            'Non-Striker': 0.2,
            'Stumps': 0.3,          # Important cricket element
            'Stumps Mic': 0.45,     # Higher threshold for less important
            'Umpire': 0.35,
            'Wicket Keeper': 0.70     # Important cricket element
        }
        
        print(f" Ready to detect {len(self.cricket_classes)} cricket objects + ball!")
        print(f" Ball detection mode: {self.ball_detection_mode}")
        print(f" Ball weightage: {self.ball_weightage} (confidence boost: {self.ball_confidence_boost})")
        print(f" Class weightage: Batting Pads = {self.class_weights['Batting Pads']} (reduced)")
        print(f" Class confidence: Batting Pads = {self.class_confidence_thresholds['Batting Pads']} (higher threshold)")
    
    def configure_class_confidence(self, class_name, confidence_threshold):
        """
        Configure confidence threshold for a specific cricket object class
        
        Args:
            class_name: Name of the cricket class (e.g., 'Batter', 'Batting Pads')
            confidence_threshold: Confidence threshold (0.1-0.9, lower = more sensitive)
        """
        if class_name in self.class_confidence_thresholds:
            self.class_confidence_thresholds[class_name] = confidence_threshold
            print(f" Confidence threshold updated: {class_name} = {confidence_threshold}")
        else:
            print(f" Class '{class_name}' not found. Available classes:")
            for cls in self.cricket_classes:
                print(f"   - {cls}")
    
    def configure_class_weight(self, class_name, weight):
        """
        Configure weightage for a specific cricket object class
        
        Args:
            class_name: Name of the cricket class (e.g., 'Batter', 'Batting Pads')
            weight: Class weight (1.0 = normal, <1.0 = less important, >1.0 = more important)
        """
        if class_name in self.class_weights:
            self.class_weights[class_name] = weight
            print(f" Class weight updated: {class_name} = {weight}")
        else:
            print(f" Class '{class_name}' not found. Available classes:")
            for cls in self.cricket_classes:
                print(f"   - {cls}")
    
    def configure_colors(self, class_name, color):
        """
        Configure color for a specific cricket object class
        
        Args:
            class_name: Name of the cricket class (e.g., 'Batter', 'Ball')
            color: RGB color tuple (e.g., (255, 0, 0) for red)
        """
        if class_name in self.cricket_colors:
            self.cricket_colors[class_name] = color
            print(f" Color updated: {class_name} = {color}")
        elif class_name == 'Ball':
            self.ball_color = color
            print(f" Ball color updated: {color}")
        else:
            print(f" Class '{class_name}' not found. Available classes:")
            for cls in self.cricket_classes + ['Ball']:
                print(f"   - {cls}")
    
    def configure_font(self, scale=0.7, thickness=2, line_spacing=25):
        """
        Configure font settings for text display
        
        Args:
            scale: Font size (0.5-2.0)
            thickness: Font thickness (1-3)
            line_spacing: Space between overlapping labels (15-40)
        """
        self.font_config['scale'] = scale
        self.font_config['thickness'] = thickness
        self.font_config['line_spacing'] = line_spacing
        print(f" Font updated: scale={scale}, thickness={thickness}, spacing={line_spacing}")
    
    def configure_ball_detection(self, mode="tracked", weightage=1.5, confidence_threshold=0.25, confidence_boost=1.2):
        """
        Configure ball detection settings
        
        Args:
            mode: "standard" or "tracked"
            weightage: Ball importance (1.0 = normal, >1.0 = more important)
            confidence_threshold: Lower = more sensitive (0.1-0.5)
            confidence_boost: Multiply confidence by this factor (1.0-2.0)
        """
        self.ball_detection_mode = mode
        self.ball_weightage = weightage
        self.ball_confidence_threshold = confidence_threshold
        self.ball_confidence_boost = confidence_boost
        
        print(f" Ball detection updated:")
        print(f"   Mode: {mode}")
        print(f"   Weightage: {weightage}")
        print(f"   Threshold: {confidence_threshold}")
        print(f"   Boost: {confidence_boost}")
    
    def find_objects_model(self):
        """Find trained cricket objects model"""
        
        model_paths = [
            "cricket_runs/cricket_objects_model/weights/best.pt",
            "cricket_runs/cricket_model_v1/weights/best.pt",
            "runs/detect/cricket_objects_model/weights/best.pt",
            "runs/detect/cricket_model_v1/weights/best.pt"
        ]
        
        for model_path in model_paths:
            if Path(model_path).exists():
                return model_path
        
        print(" No trained cricket objects model found!")
        print(" Please train your model first:")
        print("   python train_separate_models.py")
        exit(1)
    
    def find_ball_model(self):
        """Find or use ball detection model"""
        
        # First check for custom ball model
        ball_model_paths = [
            "cricket_runs/cricket_ball_model/weights/best.pt",
            "runs/detect/cricket_ball_model/weights/best.pt"
        ]
        
        for model_path in ball_model_paths:
            if Path(model_path).exists():
                print(f"   Using custom ball model: {model_path}")
                return model_path
        
        # Fallback to general YOLO model (detects sports balls)
        print("   Using general YOLO model for ball detection")
        return 'yolov8n.pt'  # General model can detect sports balls
    
    def detect_in_frame(self, frame):
        """
        Detect cricket objects AND ball in a single frame using separate models
        Combines results from both models without class conflicts
        Uses enhanced ball detection with lower threshold and multiple passes
        
        Args:
            frame: OpenCV image array (BGR format)
            
        Returns:
            List of detection dictionaries with class, confidence, box coordinates
        """
        
        detections = []
        
        # 1. Detect cricket objects using cricket objects model
        cricket_results = self.objects_model(frame)
        
        for result in cricket_results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if confidence > 0.3 and class_id < len(self.cricket_classes):
                        # Apply class-specific weightage and confidence systems
                        class_name = self.cricket_classes[class_id]
                        class_weight = self.class_weights.get(class_name, 1.0)
                        class_confidence_threshold = self.class_confidence_thresholds.get(class_name, 0.3)
                        
                        # Check class-specific confidence threshold first
                        if confidence < class_confidence_threshold:
                            continue  # Skip detections below class-specific threshold
                        
                        # Apply class weight-based threshold adjustment
                        required_threshold = class_confidence_threshold / class_weight
                        if confidence < required_threshold:
                            continue  # Skip detections that don't meet weighted threshold
                        
                        # Apply visual adjustments based on weight
                        display_confidence = confidence * class_weight
                        display_color = self.cricket_colors[class_name]  # Get color by class name
                        display_class = class_name
                        
                        # Visual modifications for low-weight classes
                        if class_weight < 0.8:
                            display_color = tuple(int(c * 0.7) for c in display_color)  # Darker color
                            display_class = f"({class_name})"  # Add parentheses for low priority
                        
                        detections.append({
                            'class': display_class,
                            'confidence': min(display_confidence, 0.99),  # Cap at 0.99
                            'box': (x1, y1, x2, y2),
                            'color': display_color,
                            'type': 'cricket_object',
                            'weight': class_weight
                        })
        
        # 2. Enhanced ball detection with multiple strategies
        ball_detections = self.detect_ball_enhanced(frame)
        detections.extend(ball_detections)
        
        return detections
    
    def detect_ball_enhanced(self, frame):
        """
        Single configurable ball detection method
        Mode: 'standard' or 'tracked'
        """
        ball_detections = []
        
        if self.ball_detection_mode == "standard":
            # Simple ball detection without tracking
            ball_detections = self.detect_ball_standard(frame)
            
        elif self.ball_detection_mode == "tracked":
            # Ball detection with tracking (recommended)
            ball_detections = self.detect_ball_tracked(frame)
        
        # Apply ball weightage to all detections
        for ball in ball_detections:
            # Apply confidence boost based on ball weightage
            ball['confidence'] = min(ball['confidence'] * self.ball_confidence_boost, 0.99)
            # Keep clean ball label without weightage display
            if 'Ball' not in ball['class']:
                ball['class'] = 'Ball'
        
        return ball_detections
    
    def detect_ball_standard(self, frame):
        """Standard ball detection without tracking"""
        ball_detections = []
        ball_results = self.ball_model(frame, conf=self.ball_confidence_threshold)
        
        for result in ball_results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id == 32 and confidence > self.ball_confidence_threshold:
                        ball_detections.append({
                            'class': 'Ball',
                            'confidence': confidence,
                            'box': (x1, y1, x2, y2),
                            'color': self.ball_color,
                            'type': 'ball'
                        })
        
        return ball_detections
    
    def detect_ball_tracked(self, frame):
        """Ball detection with tracking (recommended)"""
        ball_detections = []
        ball_results = self.ball_model(frame, conf=self.ball_confidence_threshold)
        
        for result in ball_results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id == 32 and confidence > self.ball_confidence_threshold:
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        # Apply tracking logic if we have a previous position
                        if self.last_ball_position:
                            last_x, last_y = self.last_ball_position
                            distance = ((center_x - last_x) ** 2 + (center_y - last_y) ** 2) ** 0.5
                            
                            # Boost confidence for balls near last position
                            if distance <= self.max_tracking_distance:
                                tracking_boost = max(0.1, 1.0 - (distance / self.max_tracking_distance) * 0.3)
                                confidence *= tracking_boost
                        
                        ball_detections.append({
                            'class': 'Ball (Tracked)',
                            'confidence': confidence,
                            'box': (x1, y1, x2, y2),
                            'color': self.ball_color,
                            'type': 'ball'
                        })
        
        # Update tracking information
        if ball_detections:
            best_ball = max(ball_detections, key=lambda x: x['confidence'])
            x1, y1, x2, y2 = best_ball['box']
            self.last_ball_position = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.ball_tracking_frames = 0
        else:
            self.ball_tracking_frames += 1
            if self.ball_tracking_frames > 30:  # Reset after 1 second at 30 FPS
                self.last_ball_position = None
        
        return ball_detections
    
    def draw_detections(self, frame, detections):
        """
        Draw detection boxes and labels on frame with improved font handling and no overlaps
        
        Args:
            frame: OpenCV image array to draw on
            detections: List of detection dictionaries from detect_in_frame()
            
        Returns:
            frame: Modified frame with bounding boxes and labels drawn
        """
        
        # Sort detections by confidence (highest first) to prioritize important detections
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Track used label positions to prevent overlaps
        used_positions = []
        
        # Draw each detection on the frame
        for det in sorted_detections:
            x1, y1, x2, y2 = det['box']                 # Extract bounding box coordinates
            color = det['color']                          # Get unique color for this object class
            label = f"{det['class']} {det['confidence']:.2f}"    # Create label text with confidence
            
            # Draw bounding box rectangle around detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)    # 2-pixel thick colored rectangle
            
            # Calculate label background size using font configuration
            font = self.font_config['font']
            scale = self.font_config['scale']
            thickness = self.font_config['thickness']
            
            (label_width, label_height), baseline = cv2.getTextSize(
                label, font, scale, thickness
            )
            
            # Find non-overlapping position for label
            label_y = self.find_non_overlapping_position(
                x1, y1, label_width, label_height, used_positions
            )
            
            # Ensure label stays within frame bounds
            label_x = max(0, min(x1, frame.shape[1] - label_width))
            label_y = max(label_height + 5, min(label_y, frame.shape[0] - 5))
            
            # Add transparency effect to background (if configured)
            if self.font_config.get('background_alpha', 1.0) < 1.0:
                # Create overlay for semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(
                    overlay, 
                    (label_x - 2, label_y - label_height - 8),
                    (label_x + label_width + 2, label_y + 2), 
                    color, -1
                )
                alpha = self.font_config['background_alpha']
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                # Draw solid background rectangle
                cv2.rectangle(
                    frame, 
                    (label_x - 2, label_y - label_height - 8),
                    (label_x + label_width + 2, label_y + 2), 
                    color, -1
                )
            
            # Draw white text with configured font settings
            cv2.putText(
                frame, label, (label_x, label_y - 5),
                font, scale, (255, 255, 255), thickness
            )
            
            # Record this position as used
            used_positions.append({
                'x1': label_x - 2,
                'y1': label_y - label_height - 8,
                'x2': label_x + label_width + 2,
                'y2': label_y + 2
            })
        
        return frame    # Return frame with all detections drawn
    
    def find_non_overlapping_position(self, x, y, width, height, used_positions):
        """
        Find a position for label that doesn't overlap with existing labels
        
        Args:
            x, y: Preferred position
            width, height: Label dimensions
            used_positions: List of already used label positions
            
        Returns:
            int: Y coordinate for non-overlapping position
        """
        line_spacing = self.font_config['line_spacing']
        
        # Start with preferred position (above bounding box)
        test_y = y
        
        # Check for overlaps and adjust position
        for _ in range(10):  # Maximum 10 attempts to find position
            overlaps = False
            
            # Test rectangle for current position
            test_rect = {
                'x1': x - 2,
                'y1': test_y - height - 8,
                'x2': x + width + 2,
                'y2': test_y + 2
            }
            
            # Check against all used positions
            for used_rect in used_positions:
                if self.rectangles_overlap(test_rect, used_rect):
                    overlaps = True
                    break
            
            if not overlaps:
                return test_y
            
            # Move label up to avoid overlap
            test_y -= line_spacing
            
            # If we go too high, try below the bounding box
            if test_y < height + 10:
                test_y = y + 50  # Below bounding box
        
        return test_y  # Return best position found
    
    def rectangles_overlap(self, rect1, rect2):
        """
        Check if two rectangles overlap
        
        Args:
            rect1, rect2: Dictionaries with x1, y1, x2, y2 keys
            
        Returns:
            bool: True if rectangles overlap
        """
        return not (rect1['x2'] < rect2['x1'] or 
                   rect1['x1'] > rect2['x2'] or 
                   rect1['y2'] < rect2['y1'] or 
                   rect1['y1'] > rect2['y2'])
    
    def process_video(self, video_path, output_path=None, save_video=True):
        """
        Process a cricket video and show detections with optional output saving
        Like applying a trained sklearn model to new test data
        
        Args:
            video_path: Path to input video file or webcam index (0 for default camera)
            output_path: Path to save annotated output video (optional)
            save_video: Whether to save annotated video to disk
            
        Returns:
            bool: True if processing completed successfully
        """
        
        print(f"🎥 Processing video: {video_path}")
        
        # Open video source (file or webcam)
        cap = cv2.VideoCapture(str(video_path))           # Initialize video capture object
        
        # Verify video source opened successfully
        if not cap.isOpened():
            print(f" Error: Cannot open video {video_path}")
            return False                                  # Return failure status
        
        # Extract video properties for processing setup
        fps = int(cap.get(cv2.CAP_PROP_FPS))             # Frames per second of input video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # Total number of frames in video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # Frame width in pixels
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))         # Frame height in pixels
        
        print(f" Video info: {total_frames} frames at {fps} FPS ({width}x{height})")
        
        # Setup output video writer if user wants to save results
        out_writer = None                                 # Initialize video writer as None
        if save_video and output_path:                   # Only setup if saving is requested
            print(f" Saving output to: {output_path}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')      # Define video codec (MP4 format)
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Create video writer
        
        print(" Press 'q' to quit, 'space' to pause")
        
        # Initialize tracking variables for statistics
        frame_count = 0                                   # Current frame number being processed
        total_detections = 0                              # Total number of objects detected across all frames
        ball_detections = 0                               # Total ball detections across all frames
        detection_log = []                                # List to store detailed detection results
        
        # Main video processing loop
        while True:
            ret, frame = cap.read()                       # Read next frame from video source
            
            if not ret:                                   # Check if frame was read successfully
                print(" Video finished!")
                break                                     # Exit loop if no more frames
            
            frame_count += 1                              # Increment frame counter
            
            # Apply cricket object detection to current frame (like model.predict() in sklearn)
            detections = self.detect_in_frame(frame)      # Get list of detected cricket objects + ball
            total_detections += len(detections)           # Update total detection count
            
            # Count ball detections separately
            frame_ball_count = sum(1 for d in detections if d.get('type') == 'ball')
            ball_detections += frame_ball_count
            
            # Log detections with timestamp for detailed analysis
            if detections:                                        # Only log frames with detections
                detection_log.append({
                    'frame': frame_count,                         # Frame number for reference
                    'timestamp': frame_count / fps,               # Timestamp in seconds
                    'detections': [{'class': d['class'], 'confidence': d['confidence'], 'type': d.get('type', 'cricket_object')} for d in detections]  # Extract key info
                })
            
            # Draw detection boxes and labels on frame for visualization
            if detections:                                        # Only draw if objects were detected
                frame = self.draw_detections(frame, detections)   # Add bounding boxes and labels
                
                # Show current detection count on frame (separate cricket objects and ball)
                cricket_count = len(detections) - frame_ball_count
                detection_text = f"Objects: {cricket_count} | Ball: {frame_ball_count}"
                cv2.putText(
                    frame, detection_text, (10, 30),             # Position at top-left
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 # Green text, size 1, thickness 2
                )
            
            # Show frame progress information
            frame_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(
                frame, frame_text, (10, frame.shape[0] - 10),    # Position at bottom-left
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2   # White text for visibility
            )
            
            # Save current frame to output video file (if saving enabled)
            if out_writer:                                        # Check if video writer is initialized
                out_writer.write(frame)                          # Write frame to output video file
            
            # Display frame in OpenCV window for real-time viewing
            cv2.imshow('Cricket Detection', frame)               # Show frame with detections
            
            # Handle user keyboard input for interaction
            key = cv2.waitKey(1) & 0xFF                          # Get pressed key (1ms timeout)
            if key == ord('q'):                                  # 'q' key pressed
                print(" Stopped by user")
                break                                            # Exit processing loop
            elif key == ord(' '):                                # Space key pressed
                print("  Paused - press any key to continue")
                cv2.waitKey(0)                                   # Wait indefinitely for any key
        
        # Cleanup resources after processing completion
        cap.release()                                            # Release video capture object
        if out_writer:                                           # If video writer was used
            out_writer.release()                                 # Release video writer object
        cv2.destroyAllWindows()                                  # Close all OpenCV windows
        
        # Save detailed detection log to text file for analysis
        if output_path:                                          # Only save if output path specified
            log_path = output_path.replace('.mp4', '_detections.txt')    # Create log filename
            self.save_detection_log(detection_log, log_path, total_detections, total_frames, ball_detections)  # Save results
        
        # Display final processing summary
        print(f" Finished processing {video_path}")
        print(f" Total detections: {total_detections} in {frame_count} frames")
        print(f" Cricket objects: {total_detections - ball_detections} | Ball detections: {ball_detections}")
        
        return True                                              # Return success status
    
    def save_detection_log(self, detection_log, log_path, total_detections, total_frames, ball_detections=0):
        """
        Save detection results to a text file for detailed analysis
        Includes separate tracking for cricket objects and ball detections
        
        Args:
            detection_log: List of detection dictionaries with frame info
            log_path: Path to save the log file
            total_detections: Total number of objects detected
            total_frames: Total number of frames processed
            ball_detections: Number of ball detections
        """
        
        # Write comprehensive detection report to text file
        with open(log_path, 'w') as f:
            # Header section with project information
            f.write(" Separate Cricket Detection Results\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics section
            f.write(f" Summary:\n")
            f.write(f"   Total Frames: {total_frames}\n")
            f.write(f"   Total Detections: {total_detections}\n")
            f.write(f"   Cricket Objects: {total_detections - ball_detections}\n")
            f.write(f"   Ball Detections: {ball_detections}\n")
            f.write(f"   Frames with Detections: {len(detection_log)}\n")
            f.write(f"   Detection Rate: {len(detection_log)/total_frames*100:.1f}%\n\n")
            
            # Model information
            f.write(" Detection Models:\n")
            f.write("   Cricket Objects: Custom trained model\n")
            f.write("   Ball Detection: General YOLO model\n\n")
            
            # Detailed detection results section
            f.write(" Detailed Detections:\n")
            f.write("-" * 30 + "\n")
            
            # Log each frame with detections
            for entry in detection_log:
                f.write(f"Frame {entry['frame']} ({entry['timestamp']:.1f}s):\n")
                for det in entry['detections']:
                    det_type = det.get('type', 'cricket_object')
                    f.write(f"  - {det['class']}: {det['confidence']:.2f} ({det_type})\n")
                f.write("\n")
        
        print(f" Detection log saved: {log_path}")

def find_cricket_videos():
    """
    Find available cricket video files in the current directory for testing
    
    Returns:
        list: List of video file paths found in current directory
    """
    
    # Define common video file extensions to search for
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    videos = []                                              # List to store found video files
    
    # Search for video files with each extension
    for ext in video_extensions:                             # Check each video format
        videos.extend(glob.glob(ext))                        # Find files matching extension pattern
    
    return videos                                            # Return list of available video files
    

def get_user_input():
    """
    Get input video method selection from user with interactive menu
    
    Returns:
        int: User's choice (1=file path, 2=available videos)
    """
    
    # Display input method options to user
    print(" Enter video input method:")
    print("   1. Enter video file path")                     # Manual file path entry
    print("   2. Choose from available videos")             # Select from found videos
    
    # Get valid user choice with input validation
    while True:
        try:
            choice = int(input("\nSelect option (1-2): "))   # Get user input as integer
            if choice in [1, 2]:                             # Validate choice is in valid range
                return choice                                # Return valid choice
            else:
                print(" Please enter 1 or 2")               # Error message for invalid range
        except ValueError:                                   # Handle non-integer input
            print(" Please enter a valid number")

def get_video_path_from_user():
    """
    Get video file path from user input with validation
    
    Returns:
        str: Valid video file path, or None if user cancels
    """
    
    # Loop until valid path provided or user cancels
    while True:
        video_path = input("\n Enter video file path: ").strip()    # Get file path from user
        
        if not video_path:                                   # Check if empty input
            print(" Please enter a valid path")
            continue                                         # Ask again
            
        video_path = Path(video_path)                        # Convert to Path object for validation
        
        if video_path.exists():                              # Check if file exists
            return str(video_path)                           # Return valid path as string
        else:
            print(f" File not found: {video_path}")        # Error message for missing file
            retry = input(" Try again? (y/n): ").strip().lower()    # Ask if user wants to retry
            if retry != 'y':                                 # If user doesn't want to retry
                return None                                  # Return None to indicate cancellation

def get_output_settings():
    """
    Get output settings from user for saving results
    
    Returns:
        tuple: (save_video: bool, output_path: str or None)
    """
    
    print("\n Output Settings:")
    
    # Ask if user wants to save annotated video
    while True:
        save_video = input(" Save annotated video? (y/n): ").strip().lower()    # Get save preference
        if save_video in ['y', 'yes']:                      # Accept multiple yes formats
            save_video = True
            break                                            # Exit loop with True
        elif save_video in ['n', 'no']:                     # Accept multiple no formats
            save_video = False
            break                                            # Exit loop with False
        else:
            print(" Please enter 'y' or 'n'")             # Error for invalid input
    
    # Get output file path if saving is enabled
    output_path = None                                       # Default to no output path
    if save_video:                                           # Only ask for path if saving
        default_output = "cricket_detection_output.mp4"     # Default filename
        output_path = input(f" Output file name (default: {default_output}): ").strip()
        if not output_path:                                  # If no input provided
            output_path = default_output                     # Use default filename
        
        # Ensure output file has .mp4 extension
        if not output_path.endswith('.mp4'):                 # Check file extension
            output_path += '.mp4'                            # Add .mp4 if missing
    
    return save_video, output_path                           # Return tuple of settings

def main():
    """
    Main function - Interactive cricket detection script
    Like running a trained sklearn model on new data with user interaction
    
    This function orchestrates the complete detection workflow:
    1. Load trained models
    2. Get user input (video file source)
    3. Get output preferences 
    4. Process video with ball + object detection
    5. Save results if requested
    """
    
    # Display welcome message and project information
    print(" Separate Cricket Detection")
    print("=" * 50)
    print(" Using TWO separate YOLO models:")               # Emphasize separate models approach
    print(" 1. Cricket Objects Model (12 classes)")        # Custom trained model
    print(" 2. Ball Detection Model (general)")             # General or custom ball model
    print("-" * 50)
    
    # Initialize detector by loading both models (like loading saved sklearn models)
    try:
        detector = SeparateCricketDetector()                 # Create detector instance with both models
    except SystemExit:                                       # Handle case where models not found
        return                                               # Exit if model loading failed
    
    # Get user's preferred input method (file, available videos, or webcam)
    choice = get_user_input()                                # Display menu and get user choice
    
    # Handle different input methods based on user selection
    if choice == 1:
        # Option 1: User manually enters video file path
        video_path = get_video_path_from_user()              # Get file path with validation
        if not video_path:                                   # Check if user cancelled or invalid path
            print(" No valid video path provided")
            return                                           # Exit if no valid path
            
    elif choice == 2:
        # Option 2: Choose from automatically discovered video files
        videos = find_cricket_videos()                       # Search for video files in current directory
        
        if not videos:                                       # Check if any videos were found
            print(" No video files found!")
            print(" Put your cricket videos (.mp4, .avi, .mov, .mkv) in this folder")
            return                                           # Exit if no videos available
        
        # Display available videos for user selection
        print(f"\n Found {len(videos)} video(s):")
        for i, video in enumerate(videos, 1):               # Number videos starting from 1
            print(f"   {i}. {video}")                       # Show numbered list of videos
        
        # Get user's video selection with validation
        while True:
            try:
                choice_idx = int(input(f"\nSelect video (1-{len(videos)}): ")) - 1    # Convert to 0-based index
                if 0 <= choice_idx < len(videos):           # Validate selection is in range
                    video_path = videos[choice_idx]          # Get selected video path
                    break                                    # Exit selection loop
                else:
                    print(" Invalid choice!")              # Error for out-of-range selection
            except ValueError:                               # Handle non-integer input
                print(" Please enter a valid number!")
    
    # Get user preferences for saving output video and detection log
    save_video, output_path = get_output_settings()         # Ask about saving annotated video
    
    # Start the detection process
    print(f"\n Starting detection...")                    # Inform user that processing is beginning
    print(f" Input: {video_path}")                       # Show input source
    if save_video:                                           # If saving is enabled
        print(f" Output: {output_path}")                  # Show output filename
    
    # Execute the detection pipeline (like model.predict() on test data)
    success = detector.process_video(video_path, output_path, save_video)    # Run detection on video
    
    # Display final results and cleanup
    if success:                                              # Check if processing completed successfully
        print("\n Separate model detection completed successfully!")     # Success message
        if save_video:                                       # If video was saved
            print(f" Annotated video saved: {output_path}")    # Confirm video saved
            log_path = output_path.replace('.mp4', '_detections.txt')    # Create log filename
            print(f" Detection log saved: {log_path}")     # Confirm log saved
    else:
        print("\n Detection failed!")                     # Error message if processing failed
    
    print(" To detect another video, run this script again")    # Helpful message for next use

# Script execution entry point (only run if script is executed directly)
if __name__ == "__main__":
    main()                                                   # Run main function when script is executed
