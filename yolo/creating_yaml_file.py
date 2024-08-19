import yaml
import os
import pandas as pd

dataDir = r'sl_data/' # css-data is the unzip path of the dataset
workingDir = '.' # Working Dir in google colab
df = pd.read_csv(r'c:\Users\raja\Downloads\projects\sl_projects\Datasets\Capstone 1\Part 1\labels.csv',names=['image_name','label','x1','y1','x2','y2'])
classes = sorted(list(set(df['label'])))
print('classes',classes,df.columns)
num_classes = len(classes)

file_dict = {
    'train': os.path.join(dataDir, 'train'),
    'val': os.path.join(dataDir, 'val'),
    'test': os.path.join(dataDir, 'test'),
    'nc': num_classes,
    'names': classes
}

with open(os.path.join(workingDir, 'data.yaml'), 'w+') as f:
  yaml.dump(file_dict, f)