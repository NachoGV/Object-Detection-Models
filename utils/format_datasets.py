import os
import cv2
import json
import torch
import shutil
from tqdm import tqdm
from data_generator import CocoDetection
from transformers import DetrImageProcessor, DetrForObjectDetection
from constants import CAT_IMAGES, CAT_ANNOTATIONS, ALL_IMAGES, ALL_ANNOTATIONS, TRAIN_COCO, VAL_COCO, TEST_COCO, DETR_OUT, YOLO_IMGS, YOLO_ANOS, YOLO_LABELS, YOLO_YAML, LABELS, TRAIN_YOLO_IMGS, TRAIN_YOLO_ANOS, VAL_YOLO_IMGS, VAL_YOLO_ANOS, TEST_YOLO_IMGS, TEST_YOLO_ANOS

def data2all():

    # Create Directories
    os.makedirs(ALL_IMAGES, exist_ok=True)
    os.makedirs(ALL_ANNOTATIONS, exist_ok=True)

    # Copy images to the directory
    print('Copying images...')
    for label_dir in tqdm(os.listdir(CAT_IMAGES)):
        for img in os.listdir(f'{CAT_IMAGES}/{label_dir}'):
            shutil.copy(f'{CAT_IMAGES}/{label_dir}/{img}', f'{ALL_IMAGES}/{img}')

    # Copy annotations to the directory
    print('Copying annotations...')
    for label_dir in tqdm(os.listdir(CAT_ANNOTATIONS)):
        for anno in os.listdir(f'{CAT_ANNOTATIONS}/{label_dir}'):
            shutil.copy(f'{CAT_ANNOTATIONS}/{label_dir}/{anno}', f'{ALL_ANNOTATIONS}/{anno}')

    print('\nNumber of images: ', len(os.listdir(ALL_IMAGES)))
    print('Number of annotations: ', len(os.listdir(ALL_ANNOTATIONS)))


# -----------------------------------------------------------------------------------------------
    
def img2jpg():

    print("Converting all images to JPG...")

    print(ALL_IMAGES)
    for img in tqdm(os.listdir(ALL_IMAGES)):
        name = img.split(".")[0]
        format = img.split(".")[1]
        if format != "jpg":
            os.rename(f'{ALL_IMAGES}/{img}', f'{ALL_IMAGES}/{name}.jpg')

    print(CAT_IMAGES)
    for label_dir in tqdm(os.listdir(CAT_IMAGES)):
        for img in os.listdir(f'{CAT_IMAGES}/{label_dir}'):
            name = img.split(".")[0]
            format = img.split(".")[1]
            if format != "jpg":
                os.rename(f'{CAT_IMAGES}/{label_dir}/{img}', f'{CAT_IMAGES}/{label_dir}/{name}.jpg')

    print(ALL_ANNOTATIONS)
    for anno in tqdm(os.listdir(ALL_ANNOTATIONS)):
        name = anno.split(".")[0]
        format = anno.split(".")[1]
        if format != "jpg":
            os.rename(f'{ALL_ANNOTATIONS}/{anno}', f'{ALL_ANNOTATIONS}/{name}.jpg.txt')

    print(CAT_ANNOTATIONS)
    for label_dir in tqdm(os.listdir(CAT_ANNOTATIONS)):
        for anno in os.listdir(f'{CAT_ANNOTATIONS}/{label_dir}'):
            name = anno.split(".")[0]
            format = anno.split(".")[1]
            if format != "jpg":
                os.rename(f'{CAT_ANNOTATIONS}/{label_dir}/{anno}', f'{CAT_ANNOTATIONS}/{label_dir}/{name}.jpg.txt')

# -----------------------------------------------------------------------------------------------
                
def data2YOLO():

    # Define Splits
    print('Splitting...')

    train_list = []
    val_list = []
    test_list = []
    for subf in os.listdir(CAT_IMAGES):
        subdir = os.path.join(CAT_IMAGES, subf)
        imgs = os.listdir(subdir)
        train_list.extend(imgs[:250])
        val_list.extend(imgs[250:400])
        test_list.extend(imgs[400:])

    # Create Directories
    print('Creating directories...')

    os.makedirs(YOLO_IMGS, exist_ok=True)
    os.makedirs(YOLO_ANOS, exist_ok=True)
    os.makedirs(YOLO_LABELS, exist_ok=True)

    os.makedirs(TRAIN_YOLO_IMGS, exist_ok=True)
    os.makedirs(TRAIN_YOLO_ANOS, exist_ok=True)

    os.makedirs(VAL_YOLO_IMGS, exist_ok=True)
    os.makedirs(VAL_YOLO_ANOS, exist_ok=True)

    os.makedirs(TEST_YOLO_IMGS, exist_ok=True)
    os.makedirs(TEST_YOLO_ANOS, exist_ok=True)
    
    # Copy images to the directory
    print('Copying images...')
    for img in tqdm(os.listdir(ALL_IMAGES)):
        
        if img in train_list:
            shutil.copy(f'{ALL_IMAGES}/{img}', f'{TRAIN_YOLO_IMGS}/{img}')
        elif img in val_list:
            shutil.copy(f'{ALL_IMAGES}/{img}', f'{VAL_YOLO_IMGS}/{img}')
        elif img in test_list:
            shutil.copy(f'{ALL_IMAGES}/{img}', f'{TEST_YOLO_IMGS}/{img}')

    # Copy annotations to the directory
    print('Formatting annotations...')
    for anno in tqdm(os.listdir(ALL_ANNOTATIONS)):
        
        img = anno.split('.')[0] + '.jpg'

        if img in train_list:
            with open(f'{ALL_ANNOTATIONS}/{anno}', 'r') as f:
                lines = f.readlines()[1:]
                with open(f'{TRAIN_YOLO_ANOS}/{anno}', 'w') as f:
                    for l in lines:
                        f.write(f'{LABELS[l.split(" ")[0]]} {l.split(" ")[1]} {l.split(" ")[2]} {l.split(" ")[3]} {l.split(" ")[4]}\n')

        elif img in val_list:
            with open(f'{ALL_ANNOTATIONS}/{anno}', 'r') as f:
                lines = f.readlines()[1:]
                with open(f'{VAL_YOLO_ANOS}/{anno}', 'w') as f:
                    for l in lines:
                        f.write(f'{LABELS[l.split(" ")[0]]} {l.split(" ")[1]} {l.split(" ")[2]} {l.split(" ")[3]} {l.split(" ")[4]}\n')

        elif img in test_list:
            with open(f'{ALL_ANNOTATIONS}/{anno}', 'r') as f:
                lines = f.readlines()[1:]
                with open(f'{TEST_YOLO_ANOS}/{anno}', 'w') as f:
                    for l in lines:
                        f.write(f'{LABELS[l.split(" ")[0]]} {l.split(" ")[1]} {l.split(" ")[2]} {l.split(" ")[3]} {l.split(" ")[4]}\n')

    # Create yaml file
    print('Creating yaml file...')
    with open(YOLO_YAML, 'w') as f:
        f.write(f'path: ../ExDark_YOLO\n')
        f.write(f'train: images/train\n')
        f.write(f'val: images/val\n')
        f.write(f'test: images/test\n')
        f.write(f'names:\n')
        for k, v in LABELS.items():
            f.write(f'  {v}: {k}\n')

    # Summary
    print('\nNumber of images: ', len(os.listdir(ALL_IMAGES)))
    print('Number of annotations: ', len(os.listdir(ALL_ANNOTATIONS)))
    print('\nNumber of train images: ', len(os.listdir(TRAIN_YOLO_IMGS)))
    print('Number of train annotations: ', len(os.listdir(TRAIN_YOLO_ANOS)))
    print('\nNumber of val images: ', len(os.listdir(VAL_YOLO_IMGS)))
    print('Number of val annotations: ', len(os.listdir(VAL_YOLO_ANOS)))
    print('\nNumber of test images: ', len(os.listdir(TEST_YOLO_IMGS)))
    print('Number of test annotations: ', len(os.listdir(TEST_YOLO_ANOS)))


# -----------------------------------------------------------------------------------------------

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
    json.dump(train_json, open(TRAIN_COCO, 'w'))
    json.dump(val_json, open(VAL_COCO, 'w'))
    json.dump(test_json, open(TEST_COCO, 'w'))

    print('Done!')

# -----------------------------------------------------------------------------------------------

def dert_results2COCO(enhancement):
    name = None
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

    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Paths
    model_path = DETR_OUT + '/' + name

    # Initialization
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    test_set = CocoDetection(image_directory_path=ALL_IMAGES,  annotation_file_path=TEST_COCO, image_processor=image_processor, enhancement=enhancement)
    model = DetrForObjectDetection.from_pretrained(model_path).to(device)
    results = {"annotations": []}
    
    # Loop
    image_id = 0
    for i in tqdm(range(len(test_set))):

        # Original Image
        _, anno = test_set[i]
        file_name = test_set.coco.loadImgs(i)[0]['file_name']
        original_size = anno['orig_size']
        cv_image = cv2.imread(os.path.join(ALL_IMAGES, file_name))
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

if __name__ == "__main__":
    function = input("Select function:\n  1 - img2jpg\n  2 - data2YOLO\n  3 - data2COCO\n  4 - dert_results2COCO\n>")
    if function == "1":
        img2jpg()
    elif function == "2":
        data2YOLO()
    elif function == "3":
        data2COCO()
    elif function == "4":
        inp = input("Enter enhancement:\n  1. Clahe\n  2. Color Balance Adjustement\n  3. Histogram Equialization\n  4. No Enhancement\n>")
        if inp == "1":
            dert_results2COCO('1')
        elif inp == "2":
            dert_results2COCO('2')
        elif inp == "3":
            dert_results2COCO('3')
        elif inp == "4":
            dert_results2COCO('4')
        else:
            print("Invalid enhancement")
    else:
        print("Invalid function")