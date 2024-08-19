import os
import shutil
import random
import csv
from collections import defaultdict
import pandas as pd
from normalize_bbox import normalize_bounding_boxes
# Set the paths
dataset_dir = r'SL_dataset'
train_dir = 'sl_data/train'
val_dir = 'sl_data/val'
test_dir = 'sl_data/test'

annotations_csv = os.path.join(dataset_dir, 'labels.csv')  # Assume the CSV is here

df = pd.read_csv(annotations_csv,names=['image_name','label','x1','y1','x2','y2'])
unique_labels = sorted(list(set(df['label'])))
print('classes',unique_labels,df.columns)
num_classes = len(unique_labels)

# Create the train and val directories
os.makedirs(train_dir + '/images', exist_ok=True)
os.makedirs(train_dir + '/labels', exist_ok=True)
os.makedirs(val_dir + '/images', exist_ok=True)
os.makedirs(val_dir + '/labels', exist_ok=True)
os.makedirs(test_dir + '/images', exist_ok=True)
os.makedirs(test_dir + '/labels', exist_ok=True)

# Read annotations CSV and group by image
annotations = defaultdict(list)
with open(annotations_csv, 'r') as file:
    reader = csv.reader(file)  # Adjust delimiter if needed
    for row in reader:
        # print(row)
        # print('row',row[0].split(',')[0])
        image_name, obj_class, x1, y1, x2, y2 = row
        image_name = image_name.zfill(8)+'.jpg'
        annotations[image_name].append((unique_labels.index(obj_class), x1, y1, x2, y2)) 

# Get a list of all image files
image_files = list(annotations.keys())

# Shuffle the list of image files
random.shuffle(image_files)

# Split the dataset into train and validation sets
train_files = image_files[:int(0.05 * len(image_files))]
val_files = image_files[int(0.05 * len(image_files)):int(0.06 * len(image_files))]
test_files = image_files[int(0.06 * len(image_files)):int(0.07 * len(image_files))]

# Copy images and create annotations in the respective directories
def copy_files(files, dest_dir):
    for file in files:
        src_image = os.path.join(dataset_dir,'Images', file)
        dst_image = os.path.join(dest_dir, 'images', file)
        try:
            shutil.copy(src_image, dst_image)
            dst_annotation = os.path.join(dest_dir, 'labels', os.path.splitext(file)[0] + '.txt')
            annotation_content = annotations[file]
            normalize_bounding_boxes(src_image, annotation_content, dst_annotation)

        except Exception as E:
            print(E)

copy_files(train_files, train_dir)
copy_files(val_files, val_dir)
copy_files(test_files, test_dir)
