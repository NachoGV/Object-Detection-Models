import os
import cv2
import json
import torch
import shutil
from tqdm import tqdm
from constants import *
from datasets_generators import CocoDetection
from transformers import DetrImageProcessor, DetrForObjectDetection

def data2COCO():
    # Initialize COCO json format
    train_json = {
        "images": [],
        "annotations": [],
        "categories": [],
        }

    val_json = {
        "images": [],
        "annotations": [],
        "categories": [],
        }

    test_json = {
        "images": [],
        "annotations": [],
        "categories": [],
        }

    print('Generating labels...')

    # Categories (Subfolders)
    categories = os.listdir(CAT_IMAGES)
    for i, c in enumerate(categories):
        train_json["categories"].append({"id": i, 
                                        "name": c})
        val_json["categories"].append({"id": i,
                                        "name": c})
        test_json["categories"].append({"id": i,
                                        "name": c})

    print('Splitting...')

    # Split
    train_list = []
    val_list = []
    test_list = []
    for subf in os.listdir(CAT_IMAGES):
        subdir = os.path.join(CAT_IMAGES, subf)
        imgs = os.listdir(subdir)
        train_list.extend(imgs[:250])
        val_list.extend(imgs[250:400])
        test_list.extend(imgs[400:])

    # Print values
    print('  Length of train set: ', len(train_list))
    print('  Length of val set: ', len(val_list))
    print('  Length of test set: ', len(test_list))

    print('Generating Training Set Json...')

    # Train Json
    img_id = 0
    anno_id = 0
    for img in tqdm(train_list):
        # Image
        train_json["images"].append({"id": img_id,
                                    "file_name": img})
        # Annotations
        with open(os.path.join(ALL_ANNOTATIONS, img + '.txt'), 'r') as f:
            lines = f.readlines()[1:]
            for l in lines:
                train_json['annotations'].append({"id": anno_id,
                                                "image_id": img_id,
                                                "category_id": categories.index(l.split(' ')[0]),
                                                "bbox": l.split(' ')[1:5],
                                                "area": float(l.split(' ')[3])*float(l.split(' ')[4])})   
                # Update annotation id 
                anno_id += 1
        # Update image id
        img_id += 1

    print('Generating Validation Set Json...')

    # Val Json
    img_id = 0
    anno_id = 0
    for img in tqdm(val_list):
        # Image
        val_json["images"].append({"id": img_id,
                                "file_name": img})
        # Annotations
        with open(os.path.join(ALL_ANNOTATIONS, img + '.txt'), 'r') as f:
            lines = f.readlines()[1:]
            for l in lines:
                val_json['annotations'].append({"id": anno_id,
                                                "image_id": img_id,
                                                "category_id": categories.index(l.split(' ')[0]),
                                                "bbox": l.split(' ')[1:5],
                                                "area": float(l.split(' ')[3])*float(l.split(' ')[4])})   
                # Update annotation id 
                anno_id += 1
        # Update image id
        img_id += 1

    print('Generating Test Set Json...')

    # Test Json
    img_id = 0
    anno_id = 0
    for img in tqdm(test_list):
        # Image
        test_json["images"].append({"id": img_id,
                                    "file_name": img})
        # Annotations
        with open(os.path.join(ALL_ANNOTATIONS, img + '.txt'), 'r') as f:
            lines = f.readlines()[1:]
            n_lines = len(lines)
            for l in lines:
                test_json['annotations'].append({"id": anno_id,
                                                "image_id": img_id,
                                                "category_id": categories.index(l.split(' ')[0]),
                                                "bbox": l.split(' ')[1:5],
                                                "area": float(l.split(' ')[3])*float(l.split(' ')[4]),
                                                "iscrowd": 1 if n_lines > 1 else 0})
                                                    
                # Update annotation id 
                anno_id += 1
        # Update image id
        img_id += 1

    print('Saving...')

    # Save Files
    json.dump(train_json, open(TRAIN_COCO), 'w')
    json.dump(val_json, open(VAL_COCO, 'w'))
    json.dump(test_json, open(TEST_COCO, 'w'))

    print('Done')

# -----------------------------------------------------------------------------------------------
    
def img2jpg():
    all_imgs = "../ExDark_All/Images"
    all_annos = "../ExDark_All/Annotations"
    imgs = "../ExDark/ExDark"
    annos = "../ExDark_Annno/ExDark_Annno"

    print("Converting all images to JPG...")

    print(all_imgs)
    for img in tqdm(os.listdir(all_imgs)):
        name = img.split(".")[0]
        format = img.split(".")[1]
        if format != "jpg":
            os.rename(f'{all_imgs}/{img}', f'{all_imgs}/{name}.jpg')

    print(imgs)
    for label_dir in tqdm(os.listdir(imgs)):
        for img in tqdm(os.listdir(f'{imgs}/{label_dir}')):
            name = img.split(".")[0]
            format = img.split(".")[1]
            if format != "jpg":
                os.rename(f'{imgs}/{label_dir}/{img}', f'{imgs}/{label_dir}/{name}.jpg')

    print(all_annos)
    for anno in tqdm(os.listdir(all_annos)):
        name = anno.split(".")[0]
        format = anno.split(".")[1]
        if format != "jpg":
            os.rename(f'{all_annos}/{anno}', f'{all_annos}/{name}.jpg.txt')

    print(annos)
    for label_dir in tqdm(os.listdir(annos)):
        for anno in tqdm(os.listdir(f'{annos}/{label_dir}')):
            name = anno.split(".")[0]
            format = anno.split(".")[1]
            if format != "jpg":
                os.rename(f'{annos}/{label_dir}/{anno}', f'{annos}/{label_dir}/{name}.jpg.txt')

# -----------------------------------------------------------------------------------------------

def dert_results2COCO():
    name = None
    enhancement = None
    inp = input("Enter enhancement:\n  1. Clahe\n  2. Color Balance Adjustement\n  3. Histogram Equialization\n  4. No Enhancement\n>")
    if inp == "1":
        name = "clahe/output"
        enhancement = "clahe"
    elif inp == "2":
        name = "color_balance_adjustment/output"
        enhancement = "color_balance_adjustment"
    elif inp == "3":
        name = "he/output"
        enhancement = "he"
    elif inp == "4":
        name = "raw/output"
        enhancement = 'raw'
    else:
        print("Invalid input")
        return

    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Paths
    imgs_path = "../ExDark_All/Images"
    test_path = "../ExDark_COCO/test_set.json"
    model_path = f"../Models/Transformer/lightning_logs/{name}"

    # Initialization
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    test_set = CocoDetection(image_directory_path=imgs_path,  annotation_file_path=test_path, image_processor=image_processor, enhancement=enhancement)
    model = DetrForObjectDetection.from_pretrained(model_path).to(device)
    results = {"annotations": []}
    
    # Loop
    image_id = 0
    for i in tqdm(range(len(test_set))):

        # Original Image
        _, anno = test_set[i]
        file_name = test_set.coco.loadImgs(i)[0]['file_name']
        original_size = anno['orig_size']
        cv_image = cv2.imread(os.path.join(imgs_path, file_name))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Prediction
        results_ = None
        with torch.no_grad():
            inputs = image_processor(images=cv_image, return_tensors='pt').to(device)
            outputs = model(**inputs)
            results_ = image_processor.post_process_object_detection(
                outputs=outputs, 
                threshold=0.001, 
                target_sizes=[original_size]
            )[0]

        for j in range(len(results_["scores"])):
            if results_["scores"][j] >= 0.85:
                bbox = results_["boxes"][j].tolist()
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                bbox = [float(b) for b in bbox]
                results['annotations'].append({"image_id": image_id,
                                               "category_id": results_["labels"][j].item(),
                                               "bbox": bbox,
                                               "score": results_["scores"][j].item(),})	
        image_id += 1

    # Save as json
    with open(f'../Models/Transformer/lightning_logs/{name}/results.json', 'w') as fp:
        json.dump(results, fp)

# -----------------------------------------------------------------------------------------------
        
def data2YOLO():

    cat_dir = '../ExDark/ExDark'
    images = '../ExDark_All/Images'
    annos = '../ExDark_All/Annotations'
    labels = {'Bicycle': 0, 
              'Boat': 1, 
              'Bottle': 2, 
              'Bus': 3, 
              'Car': 4, 
              'Cat': 5, 
              'Chair': 6, 
              'Cup': 7, 
              'Dog': 8, 
              'Motorbike': 9, 
              'People': 10, 
              'Table': 11}

    # Define Splits
    print('Splitting...')

    train_list = []
    val_list = []
    test_list = []
    for subf in os.listdir(cat_dir):
        subdir = os.path.join(cat_dir, subf)
        imgs = os.listdir(subdir)
        train_list.extend(imgs[:250])
        val_list.extend(imgs[250:400])
        test_list.extend(imgs[400:])

    # Create Directories
    print('Creating directories...')

    os.makedirs('../ExDark_YOLO/images', exist_ok=True)
    os.makedirs('../ExDark_YOLO/annotations', exist_ok=True)

    os.makedirs('../ExDark_YOLO/images/train', exist_ok=True)
    os.makedirs('../ExDark_YOLO/images/val', exist_ok=True)
    os.makedirs('../ExDark_YOLO/images/test', exist_ok=True)

    os.makedirs('../ExDark_YOLO/annotations/train', exist_ok=True)
    os.makedirs('../ExDark_YOLO/annotations/val', exist_ok=True)
    os.makedirs('../ExDark_YOLO/annotations/test', exist_ok=True)
    
    # Copy images to the directory
    print('Copying images...')
    for img in tqdm(os.listdir('../ExDark_All/Images')):
        
        if img in train_list:
            shutil.copy(f'{images}/{img}', f'../ExDark_YOLO/images/train/{img}')
        elif img in val_list:
            shutil.copy(f'{images}/{img}', f'../ExDark_YOLO/images/val/{img}')
        elif img in test_list:
            shutil.copy(f'{images}/{img}', f'../ExDark_YOLO/images/test/{img}')

    # Copy annotations to the directory
    print('Formatting annotations...')
    for anno in tqdm(os.listdir('../ExDark_All/Annotations')):
        
        img = anno.split('.')[0] + '.jpg'

        if img in train_list:
            with open(f'{annos}/{anno}', 'r') as f:
                lines = f.readlines()[1:]
                with open(f'../ExDark_YOLO/annotations/train/{anno}', 'w') as f:
                    for l in lines:
                        f.write(f'{labels[l.split(" ")[0]]} {l.split(" ")[1]} {l.split(" ")[2]} {l.split(" ")[3]} {l.split(" ")[4]}\n')

        elif img in val_list:
            with open(f'{annos}/{anno}', 'r') as f:
                lines = f.readlines()[1:]
                with open(f'../ExDark_YOLO/annotations/val/{anno}', 'w') as f:
                    for l in lines:
                        f.write(f'{labels[l.split(" ")[0]]} {l.split(" ")[1]} {l.split(" ")[2]} {l.split(" ")[3]} {l.split(" ")[4]}\n')

        elif img in test_list:
            with open(f'{annos}/{anno}', 'r') as f:
                lines = f.readlines()[1:]
                with open(f'../ExDark_YOLO/annotations/test/{anno}', 'w') as f:
                    for l in lines:
                        f.write(f'{labels[l.split(" ")[0]]} {l.split(" ")[1]} {l.split(" ")[2]} {l.split(" ")[3]} {l.split(" ")[4]}\n')

# -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    function = input("Select function:\n  1 - img2jpg\n  2 - data2YOLO\n  3 - data2COCO\n  4 - dert_results2COCO\n>")
    if function == "1":
        img2jpg()
    elif function == "2":
        data2YOLO()
    elif function == "3":
        data2COCO()
    elif function == "4":
        dert_results2COCO()
    else:
        print("Invalid function")