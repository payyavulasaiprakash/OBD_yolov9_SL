import cv2
import os

def normalize_bounding_boxes(image_path, lines, output_labels_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The image at {image_path} was not found.")
    
    height, width, _ = image.shape

    # # Read the labels from the text file
    # with open(labels_path, 'r') as file:
    #     lines = file.readlines()

    normalized_lines = []

    for parts in lines:
        class_id = int(parts[0])
        x_min, y_min, x_max, y_max = map(int, parts[1:])
        
        # Normalize the coordinates
        x_center = (x_min + x_max) / 2 / width
        y_center = (y_min + y_max) / 2 / height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height
        
        # Create a new normalized line
        normalized_line = f"{class_id} {x_center} {y_center} {box_width} {box_height}\n"
        normalized_lines.append(normalized_line)
    
    # Write the normalized labels to the new output file
    with open(output_labels_path, 'w') as file:
        file.writelines(normalized_lines)

# Usage example
if __name__ == "__main__":
    image_path = 'path/to/your/image.jpg'
    labels_path = 'path/to/your/labels.txt'
    output_labels_path = 'path/to/your/labels.txt'
    
    normalize_bounding_boxes(image_path, labels_path, output_labels_path)
