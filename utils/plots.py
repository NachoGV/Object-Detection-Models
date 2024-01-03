import os
import cv2
import torch
import numpy as np
import imgaug as ia
from constants import ALL_IMAGES
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw   
import torchvision.transforms as T
from transformers import DetrImageProcessor
from utils.data_generator import CocoDetection
from transformers import DetrForObjectDetection

# --------------------------------------------------------------------------------

def plot_image_vs_prediction(image_id, test_path, model_path, width, enhancement):

    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # CONSTANTS
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'lime', 'teal', 'lavender']

    # Dataset Load
    test_set = CocoDetection(image_directory_path=ALL_IMAGES,  annotation_file_path=test_path, image_processor=image_processor, enhancement=enhancement)
    id2label = {k: v["name"] for k, v in test_set.coco.cats.items()}

    # Original Image
    og_image, og_anno = test_set[image_id]
    og_labels = og_anno['class_labels']
    og_boxes = og_anno['boxes']
    og_size = og_anno['size']
    cv_image = cv2.imread(os.path.join(ALL_IMAGES, test_set.coco.loadImgs(image_id)[0]['file_name']))
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Predicted Image
    model = DetrForObjectDetection.from_pretrained(model_path).to(device)
    with torch.no_grad():
        inputs = image_processor(images=cv_image, return_tensors='pt').to(device)
        outputs = model(**inputs)
        results = image_processor.post_process_object_detection(
            outputs=outputs, 
            threshold=0.001, 
            target_sizes=[og_size]
        )[0]

    # Print Original Annotations
    print("\nGround Truth Annotations: ")
    for i in range(len(og_labels)):
        print("Label: ", id2label[int(og_labels[i])], "Box: ", [round(b,4) for b in og_boxes[i].tolist()])

    # Print Predicted Annotations only if score is above 80% confidence
    print("\nPredicted Annotations: ")
    pred_labels = []
    pred_boxes = []
    for i in range(len(results["scores"])):
        if results["scores"][i] >= 0.85:
            score = round(float(results["scores"][i]), 5)
            label = int(results['labels'][i])
            box = results["boxes"][i]
            pred_labels.append(label)
            pred_boxes.append(box)
            print("Label: ", id2label[label], "Box: ", box.tolist(), "Score: ", score)

    # Plots - Preparation
    og_image = T.ToPILImage()(og_image)
    pred_image = og_image.copy()
    og_draw = ImageDraw.Draw(og_image)
    pred_draw = ImageDraw.Draw(pred_image)

    # Plots - Start figure
    plt.figure(figsize=(15, 7.5))

    # Plots - Original
    plt.subplot(1, 2, 1)
    for i,box in enumerate(og_boxes):
        og_box = [float(b) for b in box]
        og_box = [og_box[0]*og_size[1], og_box[1]*og_size[0], og_box[2]*og_size[1], og_box[3]*og_size[0]]
        og_box = [og_box[0]-og_box[2]/2, og_box[1]-og_box[3]/2, og_box[0]+og_box[2]/2, og_box[1]+og_box[3]/2]
        og_draw.rectangle(og_box, outline=colors[og_labels[i]], width=width)
    plt.imshow(og_image)
    plt.title("Ground Truth")
    
    # Plots - Predicted
    plt.subplot(1, 2, 2)
    for i,box in enumerate(pred_boxes):
        pred_box = [float(b) for b in box]
        pred_box = [pred_box[0], pred_box[1], pred_box[2], pred_box[3]]
        pred_draw.rectangle(pred_box, outline=colors[pred_labels[i]], width=width)
    plt.imshow(pred_image)
    plt.title("Prediction")
    plt.show()

# --------------------------------------------------------------------------------

def plot_image(image_id, test_path, width, enhancement):

    # CONSTANTS
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'lime', 'teal', 'lavender']

    # Dataset Load
    test_set = CocoDetection(image_directory_path=ALL_IMAGES,  annotation_file_path=test_path, image_processor=image_processor, enhancement=enhancement)

    # Original Image
    og_image, og_anno = test_set[image_id]
    og_labels = og_anno['class_labels']
    og_boxes = og_anno['boxes']
    og_size = og_anno['size']
    cv_image = cv2.imread(os.path.join(ALL_IMAGES, test_set.coco.loadImgs(image_id)[0]['file_name']))
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Plots - Preparation
    og_image = T.ToPILImage()(og_image)
    og_draw = ImageDraw.Draw(og_image)

    # Plots - Original
    for i,box in enumerate(og_boxes):
        og_box = [float(b) for b in box]
        og_box = [og_box[0]*og_size[1], og_box[1]*og_size[0], og_box[2]*og_size[1], og_box[3]*og_size[0]]
        og_box = [og_box[0]-og_box[2]/2, og_box[1]-og_box[3]/2, og_box[0]+og_box[2]/2, og_box[1]+og_box[3]/2]
        og_draw.rectangle(og_box, outline=colors[og_labels[i]], width=width)
    plt.imshow(og_image)
    plt.axis('off')
    plt.title("Ground Truth")

# --------------------------------------------------------------------------------
    
def plot_prediction(image_id, test_path, model_path, width, enhancement):

    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # CONSTANTS
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'lime', 'teal', 'lavender']

    # Dataset Load
    test_set = CocoDetection(image_directory_path=ALL_IMAGES,  annotation_file_path=test_path, image_processor=image_processor, enhancement=enhancement)
    id2label = {k: v["name"] for k, v in test_set.coco.cats.items()}

    # Original Image
    og_image, og_anno = test_set[image_id]
    og_size = og_anno['size']
    cv_image = cv2.imread(os.path.join(ALL_IMAGES, test_set.coco.loadImgs(image_id)[0]['file_name']))
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Predicted Image
    model = DetrForObjectDetection.from_pretrained(model_path).to(device)
    with torch.no_grad():
        inputs = image_processor(images=cv_image, return_tensors='pt').to(device)
        outputs = model(**inputs)
        results = image_processor.post_process_object_detection(
            outputs=outputs, 
            threshold=0.001, 
            target_sizes=[og_size]
        )[0]

    # Predicted Annotations only if score is above 80% confidence
    pred_labels = []
    pred_boxes = []
    for i in range(len(results["scores"])):
        if results["scores"][i] >= 0.85:
            score = round(float(results["scores"][i]), 5)
            label = int(results['labels'][i])
            box = results["boxes"][i]
            pred_labels.append(label)
            pred_boxes.append(box)

    # Plots - Preparation
    og_image = T.ToPILImage()(og_image)
    pred_image = og_image.copy()
    pred_draw = ImageDraw.Draw(pred_image)

    # Plots - Predicted
    for i,box in enumerate(pred_boxes):
        pred_box = [float(b) for b in box]
        pred_box = [pred_box[0], pred_box[1], pred_box[2], pred_box[3]]
        pred_draw.rectangle(pred_box, outline=colors[pred_labels[i]], width=width)
    plt.imshow(pred_image)
    plt.axis('off')
    plt.title(f"{enhancement} Prediction")