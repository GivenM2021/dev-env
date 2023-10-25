import math
from random import randrange
import random

import os
from pathlib import Path
from PIL import Image, ImageDraw



def yolo_to_coco(center_x, center_y, box_width, box_height, im_width, im_height):
    """Returns a tupple of bbox integers in Coco format"""

    # Convert normalised values to normal pixel coordinates 
    x = eval(center_x)*im_width        #x center
    y = eval(center_y)*im_height       #y center
    w = eval(box_width)*im_width
    h = eval(box_height)*im_height

    # Top left corner
    x = int(x - (w/2) )
    y = int(y - (h/2) )

    # Coco bbox
    return (x, y, int(w), int(h) ) 

def get_testset(dataset_size, testset_percentage):
    """
    Returns a random list of samples, given total number of samples and testset percentage. 
    Usage: get_testset(int, int):
    """

    num_samples = dataset_size
    if type(dataset_size) == str:
        num_samples = eval(dataset_size)
    if type(testset_percentage) == str:
        testset_percentage = eval(testset_percentage)

    test_size = int( float(num_samples) * (testset_percentage/100) )

    return random.sample(range(num_samples), test_size)

#print( get_testset(2000, 2))


def remove_bad_samples(dataset_dir, annotation_filenames):
    clean_list = []

    #for each in tqdm(annotation_filenames):
    for i, each in enumerate(annotation_filenames):

        isSampleClean = True

        basename = os.path.basename(each)
        basename = Path(basename).stem

        imagename = os.path.join(dataset_dir, str(basename + ".jpg")  )

        if not os.path.exists(imagename):
            print(f"Skipping {imagename} since it does not exists")
            continue    

        image = None
        im_width = im_height = 0
        # Check if the image exists and can be openned 
        try:
            image = Image.open(imagename) 
            im_width, im_height = image.width, image.height
            
        except Exception as e:
            print(f"Problems loading {imagename}\n" + str(e) )
            continue    


        with open(each, "r") as f:
            #Convert to Coco json
            Lines = f.readlines()

            for bbox_id, line in enumerate(Lines):
                annotationDict ={}   

                line = line.strip()

                yolo_ann = line.split(" ")

                class_label = "smoke"

                # Center coordinates
                bbox_label, x, y, w, h = tuple(yolo_ann)
                x,y,w,h = yolo_to_coco(x,y,w,h, im_width, im_height)

                # Check if bbox is within image bounds
                if w >= im_width or h >= im_height:
                    isSampleClean = False
                    break

        #Add test file of the image, if it exists
        if isSampleClean:
            clean_list.append(each)

    return clean_list  

def draw_box(pil_draw, coco_bbox, class_idx = None):

    """coco_bbox [x,y,w,h] """ 
    x,y,w,h = coco_bbox

    if type(class_idx) == str:
        class_idx = eval(class_idx)


    r_shape = [(x, y), (x + w -1, y + h - 1)] 
    pil_draw.rectangle(r_shape, outline ="red", width=1) 

    if class_idx is not None:
        pil_draw.text((x, y), str(class_idx), fill="white")  


def make_annotation_dict(coco_bbox,class_idx, image_id, bbox_id):
    """coco_bbox [x,y,w,h] """
    x,y,w,h = coco_bbox
    x =int(x)
    y =int(y)
    w =int(w)
    h =int(h)

    if not len(coco_bbox) == 4:
        print(f"coco_bbox must have a lenght of 4")
        return None
    
    if type(class_idx) == str:
        class_idx = eval(class_idx)

    bbox_area = w*h

    # Add annotation
    annotationDict = {
        "iscrowd": 0,
        "id" : bbox_id,
        "image_id": image_id,
        "category_id": class_idx,
        "area": int(bbox_area),
        "bbox": [x,y,w,h]
    }

    return  annotationDict