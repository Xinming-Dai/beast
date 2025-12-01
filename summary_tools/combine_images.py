#!/usr/bin/env python3
"""
Combine three images from the frames folder without losing resolution.
Adds file names as labels to indicate which model they are from.
"""

from PIL import Image, ImageDraw, ImageFont
import os

def combine_images_with_labels():
    # Image paths
    frames_dir = "/u/xdai3/beast/frames"
    images = [
        "original.png",
        "beast.png",
        "beast_with_perceptual.png"
    ]
    
    # Load all images
    loaded_images = []
    for img_name in images:
        img_path = os.path.join(frames_dir, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            loaded_images.append((img, img_name))
            print(f"Loaded {img_name}: {img.size}")
        else:
            print(f"Warning: {img_path} not found")
    
    if not loaded_images:
        print("No images found to combine!")
        return
    
    # Get dimensions
    widths = [img.width for img, _ in loaded_images]
    heights = [img.height for img, _ in loaded_images]
    
    # Calculate combined dimensions (horizontal layout)
    total_width = sum(widths)
    max_height = max(heights)
    
    # Add space for labels (50 pixels at the top for each image)
    label_height = 50
    combined_height = max_height + label_height
    
    # Create new image with white background
    combined = Image.new('RGB', (total_width, combined_height), color='white')
    
    # Try to load a nice font, fallback to default if not available
    try:
        # Try to use a system font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 24)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
    
    # Paste images and add labels
    x_offset = 0
    draw = ImageDraw.Draw(combined)
    
    for img, img_name in loaded_images:
        # Extract model name from filename (remove .png extension)
        model_name = os.path.splitext(img_name)[0]
        
        # Paste image below the label area
        combined.paste(img, (x_offset, label_height))
        
        # Draw label background
        label_rect = [x_offset, 0, x_offset + img.width, label_height]
        draw.rectangle(label_rect, fill='black')
        
        # Draw text (centered in label area)
        text_bbox = draw.textbbox((0, 0), model_name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x_offset + (img.width - text_width) // 2
        text_y = (label_height - text_height) // 2
        
        draw.text((text_x, text_y), model_name, fill='white', font=font)
        
        x_offset += img.width
    
    # Save the combined image
    output_path = os.path.join(frames_dir, "combined_models.png")
    combined.save(output_path, 'PNG', quality=100)
    print(f"\nCombined image saved to: {output_path}")
    print(f"Final dimensions: {combined.size}")
    print(f"Original images preserved at full resolution")

if __name__ == "__main__":
    combine_images_with_labels()

