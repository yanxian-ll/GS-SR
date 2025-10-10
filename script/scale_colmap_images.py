import os
import argparse
import numpy as np
from PIL import Image as PILImage

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from gssr.utils.colmap_read_write_model import \
    read_model, write_model, Camera, Image, Point3D, CAMERA_MODEL_NAMES

def resize_image_and_update_camera(image_path, max_image_size, camera):
    """
    Resize image and update camera intrinsic parameters
    
    Parameters:
        image_path: input image path
        max_image_size: maximum side length
        camera: camera object
    
    Returns:
        new_image: resized image (PIL Image object)
        new_camera: updated camera object
        scale: scaling factor
    """
    # Open original image
    original_image = PILImage.open(image_path)
    original_width, original_height = original_image.size
    
    # Calculate scaling factor
    scale = min(max_image_size / original_width, max_image_size / original_height)
    
    # Calculate new dimensions
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    
    # Resize image
    new_image = original_image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
    
    # Update camera parameters
    if camera.model == "SIMPLE_PINHOLE":
        # SIMPLE_PINHOLE: [f, cx, cy]
        f, cx, cy = camera.params
        new_params = np.array([f * scale, cx * scale, cy * scale])
    elif camera.model == "PINHOLE":
        # PINHOLE: [fx, fy, cx, cy]
        fx, fy, cx, cy = camera.params
        new_params = np.array([fx * scale, fy * scale, cx * scale, cy * scale])
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}. Only SIMPLE_PINHOLE and PINHOLE are supported")
    
    # Create new camera object
    new_camera = Camera(
        id=camera.id,
        model=camera.model,
        width=new_width,
        height=new_height,
        params=new_params
    )
    
    return new_image, new_camera, scale

def process_colmap_model(input_path, output_path, max_image_size, output_format=".bin"):
    """
    Process COLMAP model: resize images and update camera parameters
    
    Parameters:
        input_path: input model path
        output_path: output model path
        max_image_size: maximum image side length
        output_format: output format (".bin" or ".txt")
    """
    # Read COLMAP model
    print(f"Reading COLMAP model from: {input_path}")
    cameras, images, points3D = read_model(os.path.join(input_path, "sparse/0/"))

    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    output_image_dir = os.path.join(output_path, "images")
    os.makedirs(output_image_dir, exist_ok=True)
    output_sparse_dir = os.path.join(output_path, "sparse/0")
    os.makedirs(output_sparse_dir, exist_ok=True)
    
    # Check camera types, exit if not ["SIMPLE_PINHOLE", "PINHOLE"]
    for camera in cameras.values():
        if camera.model not in ["SIMPLE_PINHOLE", "PINHOLE"]:
            print(f"Error: Only SIMPLE_PINHOLE and PINHOLE camera models are supported. Found unsupported model: {camera.model}")
            return

    # Initialize updated cameras and images dictionaries
    updated_cameras = {}
    updated_images = {}
    
    # Track processed cameras to avoid duplicate processing
    processed_cameras = set()
    
    # First iterate through images, process each image
    for img_id, img in images.items():
        camera_id = img.camera_id
        camera = cameras[camera_id]
        
        # Build original image path
        original_image_path = os.path.join(input_path, "images", img.name)
        
        if not os.path.exists(original_image_path):
            print(f"Warning: Image file does not exist: {original_image_path}, skipping")
            # If image doesn't exist, use original camera and image parameters
            updated_images[img_id] = img
            if camera_id not in updated_cameras:
                updated_cameras[camera_id] = camera
            continue
        
        try:
            # If this camera hasn't been processed yet, process image and update camera
            if camera_id not in processed_cameras:
                # Resize image and update camera parameters
                resized_image, new_camera, scale = resize_image_and_update_camera(
                    original_image_path, max_image_size, camera
                )
                
                # Save resized image
                output_image_path = os.path.join(output_image_dir, img.name)
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                resized_image.save(output_image_path)
                
                # Record updated camera
                updated_cameras[camera_id] = new_camera
                processed_cameras.add(camera_id)
                
                print(f"Camera {camera_id}: {camera.width}x{camera.height} -> {new_camera.width}x{new_camera.height}")
            else:
                # Camera already processed, directly use updated camera parameters
                new_camera = updated_cameras[camera_id]
                scale = new_camera.width / camera.width
                
                # Still need to save resized image (but using same scaling parameters)
                original_image = PILImage.open(original_image_path)
                resized_image = original_image.resize((new_camera.width, new_camera.height), PILImage.Resampling.LANCZOS)
                output_image_path = os.path.join(output_image_dir, img.name)
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                resized_image.save(output_image_path)
            
            # Update 2D point coordinates in image
            new_xys = img.xys * scale
            
            # Create updated image object
            updated_images[img_id] = Image(
                id=img.id,
                qvec=img.qvec,
                tvec=img.tvec,
                camera_id=img.camera_id,
                name=img.name,
                xys=new_xys,
                point3D_ids=img.point3D_ids
            )
            
        except Exception as e:
            print(f"Error processing image {img_id} ({img.name}): {e}, using original parameters")
            # On error, use original parameters
            updated_images[img_id] = img
            if camera_id not in updated_cameras:
                updated_cameras[camera_id] = camera
    
    # Ensure all cameras are included in updated_cameras
    for camera_id, camera in cameras.items():
        if camera_id not in updated_cameras:
            updated_cameras[camera_id] = camera
    
    # Write updated model
    print(f"Writing updated model to: {output_sparse_dir}")
    write_model(updated_cameras, updated_images, points3D, output_sparse_dir, ext=output_format)
    
    print("Processing completed!")
    print(f"Resized images saved to: {output_image_dir}")
    print(f"Updated model saved to: {output_sparse_dir}")

def main():
    parser = argparse.ArgumentParser(description="Resize COLMAP SfM result images and update camera intrinsics")
    parser.add_argument("--input_model", "-i", required=True, help="Input COLMAP model directory path")
    parser.add_argument("--output_model", "-o", required=True, help="Output COLMAP model directory path")
    parser.add_argument("--max_image_size", type=int, default=1600, help="Maximum image side length")
    parser.add_argument("--output_format", choices=[".bin", ".txt"], default=".bin", 
                       help="Output model format (default: .bin)")
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_model):
        print(f"Error: Input directory does not exist: {args.input_model}")
        return
    
    # Process COLMAP model
    process_colmap_model(
        input_path=args.input_model,
        output_path=args.output_model,
        max_image_size=args.max_image_size,
        output_format=args.output_format,
    )

if __name__ == "__main__":
    main()