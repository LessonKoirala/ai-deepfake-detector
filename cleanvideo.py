import os
import cv2
import numpy as np

# A function to process videos and save their frames as images.
def process_videos_to_frames(input_dir, output_dir, frame_interval=10, image_size=(224, 224)):
    """
    Processes all video files in the specified input directory and its subdirectories.
    It extracts frames, resizes them, and saves them to a new output directory
    structure, ready for model training.

    Args:
        input_dir (str): Path to the root directory containing 'real' and 'ai' video folders.
        output_dir (str): Path to the directory where processed images will be saved.
        frame_interval (int): The number of frames to skip between saving. A higher
                              value reduces the dataset size and processing time.
        image_size (tuple): The target size for the extracted frames (width, height).
    """
    print("Starting video processing...")

    # Define the classes (real and ai) and their respective subdirectories
    classes = ['ai', 'real']
    
    # Create the output directories if they don't exist
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
        print(f"Created output directory: {os.path.join(output_dir, cls)}")

    # Iterate over each class directory
    for cls in classes:
        video_path = os.path.join(input_dir, cls)
        # Check if the path exists to avoid errors
        if not os.path.exists(video_path):
            print(f"Directory not found: {video_path}. Skipping.")
            continue
            
        print(f"\nProcessing videos in the '{cls}' directory...")
        
        # Get a list of all video files in the current directory
        video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
        
        # Process each video file
        for video_file in video_files:
            video_filepath = os.path.join(video_path, video_file)
            
            # Use OpenCV to open the video file
            cap = cv2.VideoCapture(video_filepath)
            if not cap.isOpened():
                print(f"Error: Could not open video file {video_filepath}. Skipping.")
                continue

            frame_count = 0
            frames_saved = 0
            
            while True:
                # Read a frame from the video
                ret, frame = cap.read()
                
                # If a frame was not successfully read, we've reached the end of the video
                if not ret:
                    break
                
                # Only save a frame every 'frame_interval' frames to reduce redundancy
                if frame_count % frame_interval == 0:
                    try:
                        # Resize the frame to a consistent size for the model
                        resized_frame = cv2.resize(frame, image_size)
                        
                        # Define the output image file path
                        output_image_path = os.path.join(output_dir, cls, f"{os.path.splitext(video_file)[0]}_frame_{frames_saved}.jpg")
                        
                        # Save the processed frame as a JPG file
                        cv2.imwrite(output_image_path, resized_frame)
                        frames_saved += 1
                    except cv2.error as e:
                        print(f"OpenCV error processing {video_file} at frame {frame_count}: {e}")
                
                frame_count += 1
            
            # Release the video capture object
            cap.release()
            print(f"Finished processing {video_file}. Saved {frames_saved} frames.")

    print("\nVideo processing complete. All frames have been saved.")

# Main block to run the script
if __name__ == "__main__":
    # Define input and output directories
    # The input directory is the 'videodata' folder you showed me.
    input_video_data_dir = './videodata'
    # The output directory will store the processed images
    output_image_data_dir = './processed_image_data'

    # Call the function to start the video-to-frame conversion
    process_videos_to_frames(input_video_data_dir, output_image_data_dir)
