import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
import multiprocessing
from functools import partial

def load_coefficients(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def create_polynomial_features():
    poly = PolynomialFeatures(degree=2)
    # Create dummy data to fit the polynomial features
    dummy_data = np.array([[0.0], [0.5], [1.0]])
    poly.fit(dummy_data)
    return poly

def transform_image(image, coefficients, poly):
    # Convert image to float32 and normalize to [0, 1]
    img_float = image.astype(np.float32) / 255.0
    
    # Create output image
    transformed = np.zeros_like(img_float)
    
    # Transform each channel
    # OpenCV uses BGR, coefficients are in RGB order
    channel_mapping = {'R': 2, 'G': 1, 'B': 0}  # BGR to RGB mapping
    for channel, idx in channel_mapping.items():
        channel_data = img_float[:, :, idx].reshape(-1, 1)
        channel_data_poly = poly.transform(channel_data)
        
        # Apply transformation
        transformed_channel = np.dot(channel_data_poly, coefficients[channel]['coefficients']) + coefficients[channel]['intercept']
        transformed[:, :, idx] = transformed_channel.reshape(img_float.shape[0], img_float.shape[1])
    
    # Clip values to [0, 1] and convert back to uint8
    transformed = np.clip(transformed, 0, 1)
    return (transformed * 255).astype(np.uint8)

def process_single_image(args):
    filename, input_dir, output_dir, coefficients, poly = args
    # Read image
    input_path = os.path.join(input_dir, filename)
    image = cv2.imread(input_path)
    
    if image is None:
        print(f"Could not read image: {input_path}")
        return
    
    # Transform image
    transformed = transform_image(image, coefficients, poly)
    
    # Save transformed image with the same name
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, transformed)
    print(f"Processed: {filename}")

def process_directory(input_dir, output_dir, coefficients, num_processes=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize polynomial features
    poly = create_polynomial_features()
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    
    # Prepare arguments for multiprocessing
    process_args = [(f, input_dir, output_dir, coefficients, poly) for f in image_files]
    
    # Use specified number of processes or default to CPU count
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_single_image, process_args)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Transform images using color mapping coefficients')
    parser.add_argument('input', help='Input directory containing images')
    parser.add_argument('output', nargs='?', help='Output directory for transformed images (default: input directory name + "_transformed")')
    parser.add_argument('--coefficients', '-c', default='mapping_coefficients.json', 
                        help='Path to coefficients JSON file (default: mapping_coefficients.json)')
    parser.add_argument('--processes', '-p', type=int, 
                        help='Number of processes to use (default: number of CPU cores)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.output is None:
        args.output = args.input.rstrip('/\\') + '_transformed'
    
    # Load coefficients
    coefficients = load_coefficients(args.coefficients)
    
    # Process all images in the input directory
    process_directory(args.input, args.output, coefficients, args.processes)
    print("Image transformation completed!")

if __name__ == "__main__":
    main() 