
# Categorized ExDark dataset
CAT_IMAGES = '../data/ExDark/'
CAT_ANNOTATIONS = '../data/ExDark_Annno'

# ExDark dataset
ALL_IMAGES = '../data/ExDark_All/Images'
ALL_ANNOTATIONS = '../data/ExDark_All/Annotations'

# COCO Format dataset
TRAIN_COCO = '../data/ExDark_COCO/train_set.json'
VAL_COCO = '../data/ExDark_COCO/val_set.json'
TEST_COCO = '../data/ExDark_COCO/test_set.json'

# YOLO Format dataset (root)
YOLO_IMGS = '../data/ExDark_YOLO/images'
YOLO_ANOS = '../data/ExDark_YOLO/annotations'
YOLO_LABELS = '../data/ExDark_YOLO/labels'
YOLO_YAML = '../data/ExDark_YOLO/yolo.yaml'

# YOLO Format dataset (train, val, test)
TRAIN_YOLO_IMGS = '../data/ExDark_YOLO/images/train'
TRAIN_YOLO_ANOS = '../data/ExDark_YOLO/annotations/train'
VAL_YOLO_IMGS = '../data/ExDark_YOLO/images/val'
VAL_YOLO_ANOS = '../data/ExDark_YOLO/annotations/val'
TEST_YOLO_IMGS = '../data/ExDark_YOLO/images/test'
TEST_YOLO_ANOS = '../data/ExDark_YOLO/annotations/test'

# Outputs
DETR_OUT = '../outputs/transformer'
YOLO_OUT = '../outputs/yolo'

# Labels
LABELS = {'Bicycle': 0, 
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