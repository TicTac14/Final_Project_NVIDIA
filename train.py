# Create a new Google Collab file.

!git clone https://github.com/dusty-nv/pytorch-classification.git # Run this command on a single cell on the collab to download the necessary folder.

!cd pytorch-classification; pip install -r requirements.txt # Run this command on a separate cell on the collab to install the contents within the folder necessary.

# Dataset used can be accessed through this link: https://www.kaggle.com/datasets/yousefmohamed20/sentiment-images-classifier

# Upload the dataset as a .zip file to the Google Drive. Make sure it is the Google Drive with the same account that you signed into Google Collab with.

# Make a new cell
from google.colab import drive # import library to connect drive to collab
drive.mount('/content/drive') # Gathers all the data in the Google Drive and makes it accessible from Google Collab

# Make a new cell
!cd pytorch-classification/data; unzip -q /content/drive/MyDrive/archive.zip # this unzips the data in the Google Drive. 

#Make a new cell

#script to split dataset into test/train/val
import os
from math import ceil
from random import sample,choice
from shutil import move

os.chdir("/content/drive/MyDrive") # Changes directory to the Google Drive.

# Create function to get model class names
def get_model_class_names(data_directory: str) -> list:
    class_names = os.listdir(data_directory)
    return class_names
  
# Create function to create train, test, split folders
def list_class_images(data_directory: str, class_name: str, extension_filters = ('jpg', 'png', 'jpeg', 'webp')) -> dict:
    class_images = {}
    for category in ['train', 'test', 'val']:
        class_images[category] = []
        image_directory = os.path.join(data_directory, category, class_name)
        os.makedirs(image_directory, exist_ok=True)
        if not os.listdir(image_directory) and category=="train":
            os.rmdir(image_directory)
            move(os.path.join(data_directory, class_name), os.path.join(data_directory, category))
        for file_name in os.listdir(image_directory):
            name, extension = os.path.splitext(file_name)
            extension = extension.lower().lstrip(".")
            if extension in extension_filters:
                class_images[category].append(file_name)
    return class_images

# Create function to add teh images to the train, test, split folders based on train-val-test split. 
def split_class_images(data_directory: str, class_name: str, test_percent: float = 0.1, val_percent: float = 0.1):
    class_images = list_class_images(data_directory, class_name)
    total_images = sum(map(len, class_images.values()))
    print(f'There are {total_images} images of the class {class_name}.')
    test_image_count = int(ceil(test_percent * total_images))
    val_image_count = int(ceil(val_percent * total_images))
    train_image_count = total_images - test_image_count - val_image_count
    print(f'Image dataset split: Train={train_image_count}, Test={test_image_count}, Val={val_image_count}.')

    category_counts = {
        "test": test_image_count,
        "val": val_image_count
    }

    for category_name, category_count in category_counts.items():
        if len(class_images[category_name]) > category_count:
            move_image_count = len(class_images[category_name]) - category_count
            randomly_selected_images = sample(class_images[category_name], move_image_count)
            destination_folder = os.path.join(data_directory, 'train', class_name)
            source_folder = os.path.join(data_directory, category_name, class_name)
            for file_name in randomly_selected_images:
                destination_file = os.path.join(destination_folder, file_name)
                source_file = os.path.join(source_folder, file_name)
                os.rename(source_file, destination_file)
            class_images[category_name] = list(sorted(set(class_images[category_name]).difference(randomly_selected_images)))
            class_images['train'] = list(sorted(set(class_images['train']).union(randomly_selected_images)))
        elif len(class_images[category_name]) == category_count:
            print(f'No changes are necessary for class {class_name} {category_name}.')

    for category_name, category_count in category_counts.items():
        if len(class_images[category_name]) < category_count:
            move_image_count = category_count - len(class_images[category_name])
            randomly_selected_images = sample(class_images['train'], move_image_count)
            source_folder = os.path.join(data_directory, 'train', class_name)
            destination_folder = os.path.join(data_directory, category_name, class_name)
            for file_name in randomly_selected_images:
                destination_file = os.path.join(destination_folder, file_name)
                source_file = os.path.join(source_folder, file_name)
                os.rename(source_file, destination_file)
            class_images[category_name] = list(sorted(set(class_images[category_name]).union(randomly_selected_images)))
            class_images['train'] = list(sorted(set(class_images['train']).difference(randomly_selected_images)))

folder = "Emotions_for_image_classification" #replace with folder name
classes = get_model_class_names(folder)
print(f"The classes in our dataset at {folder} are: {classes}")
for class_name in classes:
    split_class_images(folder, class_name)

# Make New cell

!cd /content/pytorch-classification; python3 train.py --model-dir=models data/$folder # This will fine-tune a ResNet-18 Model to the data. This will start the training process.

# Make new cell

!pip install onnx onnxruntime # Install libraries necessary to export model to jetson.

# Make new cell

!cd /content/pytorch-classification;python3 onnx_export.py --model-dir=models # Export thde model

# Download the exported model.

# PART 2: USING THE JETSON NANO

# Now, open up VS Code and connect to the Jetson Nano.

# Drag and drop the downloaded model into "jetson-inference/python/training/classification/models"

# Download the labels.txt file in your collab and drag it into the same directory

# Download an image from your dataset and drag it into the same directory.

# Open the terminal, cd into "jetson-inference/python/training/classification/models"

# Run the command "imagenet.py --model=resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=labels.txt INSERT_IMG_NAME.jpg INSERT_IMG_NAME-inference.jpg"
# Replace INSERT_IMG_NAME with the name of the image in the directory. You may want to rename some images to make them easier to access.






