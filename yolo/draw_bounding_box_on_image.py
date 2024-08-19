import cv2

def draw_bounding_boxes(image_path, labels_path):
    # Load the image
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    if image is None:
        raise FileNotFoundError(f"The image at {image_path} was not found.")
    
    # Read the labels from the text file
    with open(labels_path, 'r') as file:
        lines = file.readlines()
    
    # Loop through each line in the file
    for line in lines:
        # Parse the line to get class_id and bounding box coordinates
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
        
        # Convert from YOLO format to pixel coordinates
        x_center, y_center = x_center * width, y_center * height
        bbox_width, bbox_height = bbox_width * width, bbox_height * height
        
        # Calculate top-left and bottom-right coordinates
        x_min = int(x_center - bbox_width / 2)
        y_min = int(y_center - bbox_height / 2)
        x_max = int(x_center + bbox_width / 2)
        y_max = int(y_center + bbox_height / 2)
        
        # Draw the bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Put the class_id text on the bounding box
        cv2.putText(image, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image

# Usage example
if __name__ == "__main__":
    image_path = r'C:\Users\raja\Downloads\projects\Object_detection\yolov9\sl_data\train\images\00000016.jpg'
    labels_path = r'C:\Users\raja\Downloads\projects\Object_detection\yolov9\sl_data\train\labels\00000016.txt'
    
    # Draw bounding boxes
    result_image = draw_bounding_boxes(image_path, labels_path)
    
    # Display the image
    cv2.imshow("Image with Bounding Boxes", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the result image (optional)
    cv2.imwrite(r'C:\Users\raja\Downloads\projects\Object_detection\yolov9\00000016_bbox_draw.jpg', result_image)
