#!/usr/bin/env python3
"""
Headless Frame Extractor for Cricket Videos
Extracts frames from cricket videos for annotation in Roboflow

This script shows how the hybrid dataset was prepared:
1. Extract frames from cricket videos
2. Save frames for manual annotation
3. Upload to Roboflow for labeling (include cricket ball + objects)
4. Download labeled dataset for hybrid training

Usage:
    python3 headless_frame_extractor.py
    python3 headless_frame_extractor.py --video cricket_match.mp4
    python3 headless_frame_extractor.py --headless --interval 30 --max-frames 100
"""

# Import required libraries
import cv2          # OpenCV for video processing and frame extraction
from pathlib import Path    # Modern path handling for cross-platform compatibility
import argparse     # Command-line argument parsing for script options

def extract_frames_from_video(video_path, output_dir, frame_interval=30, max_frames=100):
    """
    Extract frames from cricket video for annotation
    
    Args:
        video_path: Path to cricket video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (default: 30 = 1 frame per second at 30fps)
        max_frames: Maximum number of frames to extract
    """
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)                    # Convert string path to Path object
    output_path.mkdir(parents=True, exist_ok=True)    # Create directory and parent dirs if needed
    
    # Open video file using OpenCV
    cap = cv2.VideoCapture(str(video_path))           # Initialize video capture object
    
    # Check if video file opened successfully
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    # Get video properties for information display
    fps = int(cap.get(cv2.CAP_PROP_FPS))              # Frames per second of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    duration = total_frames / fps                     # Calculate video duration in seconds
    
    # Display video information to user
    print(f"   Video Info:")
    print(f"   File: {video_path}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Frame Interval: Every {frame_interval} frames")
    print(f"   Max Frames: {max_frames}")
    
    # Initialize counters for tracking progress
    frame_count = 0      # Current frame number being processed
    extracted_count = 0  # Number of frames successfully extracted
    
    print(f"\n Extracting frames to: {output_dir}")
    
    # Main loop to process video frames
    while True:
        ret, frame = cap.read()    # Read next frame from video
        
        # Break loop if no more frames are available
        if not ret:
            break
        
        # Extract frame only at specified intervals (e.g., every 30th frame)
        if frame_count % frame_interval == 0:
            # Create descriptive filename with timestamp information
            timestamp = frame_count / fps                           # Calculate timestamp in seconds
            filename = f"frame_{extracted_count:04d}_{timestamp:.1f}s.jpg"  # Format: frame_0001_5.2s.jpg
            frame_path = output_path / filename                     # Create full file path
            
            # Save the frame as JPEG image
            cv2.imwrite(str(frame_path), frame)                     # Write frame to disk
            extracted_count += 1                                    # Increment extracted frame counter
            
            print(f"   Extracted: {filename}")                   # Show progress to user
            
            # Stop extraction if maximum number of frames reached
            if extracted_count >= max_frames:
                print(f"   Reached maximum frames ({max_frames})")
                break
        
        frame_count += 1    # Increment total frame counter
    
    # Clean up video capture object
    cap.release()
    
    # Display extraction summary and next steps
    print(f"\nExtraction complete!")
    print(f"   Extracted {extracted_count} frames from {total_frames} total frames")
    print(f"   Frames saved in: {output_dir}")
    print(f"\nNext steps:")
    print(f"   1. Review extracted frames in {output_dir}")
    print(f"   2. Upload good frames to Roboflow for annotation")
    print(f"   3. Annotate cricket ball + cricket objects (bails, batter, stumps, etc.)")
    print(f"   4. Download annotated dataset")
    print(f"   5. Train hybrid model with train_custom_cricket_model.py")
    
    return True    # Return success status

def find_cricket_videos():
    """Find cricket video files in current directory"""
    
    # Define supported video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']    # Common video formats
    videos = []                                                    # List to store found video files
    
    current_dir = Path('.')    # Get current directory as Path object
    
    # Search for video files with each extension (both lowercase and uppercase)
    for ext in video_extensions:
        videos.extend(list(current_dir.glob(f"*{ext}")))          # Find lowercase extensions
        videos.extend(list(current_dir.glob(f"*{ext.upper()}")))  # Find uppercase extensions
    
    return videos    # Return list of found video files

def main():
    """Main function for headless frame extraction"""
    
    # Display welcome message and script purpose
    print("Cricket Video Frame Extractor")
    print("=" * 50)
    print("This tool extracts frames from cricket videos for annotation")
    print("")
    
    # Setup command-line argument parser for script options
    parser = argparse.ArgumentParser(description='Extract frames from cricket videos')
    parser.add_argument('--video', '-v', type=str, help='Path to cricket video file')
    parser.add_argument('--output', '-o', type=str, default='extracted_frames', 
                       help='Output directory for frames (default: extracted_frames)')
    parser.add_argument('--interval', '-i', type=int, default=30,
                       help='Frame extraction interval (default: 30 = 1 frame per second at 30fps)')
    parser.add_argument('--max-frames', '-m', type=int, default=100,
                       help='Maximum frames to extract (default: 100)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no interactive prompts)')
    
    # Parse command-line arguments provided by user
    args = parser.parse_args()
    
    # Find all available video files in current directory
    available_videos = find_cricket_videos()
    
    # Determine which video to process based on user input
    if args.video:
        # Use video specified in command-line argument
        video_path = Path(args.video)
        if not video_path.exists():                               # Check if file exists
            print(f"Error: Video file not found: {args.video}")
            return
    elif args.headless:
        # Headless mode - automatically use first available video
        if not available_videos:
            print("Error: No video files found in current directory")
            print("Supported formats: .mp4, .avi, .mov, .mkv, .wmv")
            return
        video_path = available_videos[0]                          # Use first video found
        print(f"Using video: {video_path}")
    else:
        # Interactive mode - let user choose from available videos
        if not available_videos:
            print("Error: No video files found in current directory")
            print("Put your cricket videos in this folder")
            print("Supported formats: .mp4, .avi, .mov, .mkv, .wmv")
            return
        
        # Display list of available videos for user selection
        print("Available cricket videos:")
        for i, video in enumerate(available_videos, 1):
            print(f"   {i}. {video.name}")
        
        # Get user's video selection with input validation
        while True:
            try:
                choice = int(input(f"\nSelect video (1-{len(available_videos)}): ")) - 1
                if 0 <= choice < len(available_videos):           # Validate choice is in range
                    video_path = available_videos[choice]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:                                     # Handle non-integer input
                print("Please enter a valid number.")
    
    # Create output directory name based on video name if using default
    if args.output == 'extracted_frames':
        video_name = video_path.stem                              # Get filename without extension
        output_dir = f"extracted_frames_{video_name}"             # Create unique output directory
    else:
        output_dir = args.output                                  # Use user-specified output directory
    
    # Display processing parameters to user
    print(f"\nProcessing video: {video_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"Frame interval: Every {args.interval} frames")
    print(f"Max frames: {args.max_frames}")
    
    # Ask for confirmation in interactive mode
    if not args.headless:
        confirm = input("\nStart extraction? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Extraction cancelled.")
            return
    
    # Execute frame extraction with specified parameters
    success = extract_frames_from_video(
        video_path=video_path,
        output_dir=output_dir,
        frame_interval=args.interval,
        max_frames=args.max_frames
    )
    
    # Display final results and next steps
    if success:
        print(f"\nFrame extraction completed successfully!")
        print(f"\nData Preparation Workflow:")
        print(f"   1. Extract frames from video → {output_dir}/")
        print(f"   2. Upload frames to Roboflow for annotation")
        print(f"   3. Annotate cricket ball + cricket objects manually")
        print(f"   4. Download labeled dataset as 'roboflow_dataset/'")
        print(f"   5. Train hybrid model: python3 train_custom_cricket_model.py")
    else:
        print("Frame extraction failed.")

# Script entry point - run main function when script is executed directly
if __name__ == "__main__":
    main()
