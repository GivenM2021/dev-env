from tqdm import tqdm

from PIL import Image, ImageDraw

import os, sys
import json
import glob
from time import sleep
from pathlib import Path

import albumentations
import numpy as np

from yolo_db_functions import get_testset, remove_bad_samples, yolo_to_coco

from yolo_db_functions import make_annotation_dict, draw_box

ENABLE_VERBOSE_INFO = False
ENABLE_BBOX_AUGMENTATION = True
ENABLE_BBOX_VISUALISATION = True

testset_percentage = 15

num_augmentation_replicas = 6                   #per image

# abs_dataset_path = "/home/earlywarn/dataset/vac_student_smoke/custom_data/"
abs_dataset_path = "./annotations"

dataset_basename = os.path.basename(abs_dataset_path)
dataset_basename = Path(dataset_basename).stem

dataset_dir = abs_dataset_path #os.path.join( os.curdir, "annotations" )

print(f"Dataset Directory: {dataset_dir}")

# Search annotation files
annotation_filenames = glob.glob(os.path.join(dataset_dir, "*.txt") )


# Save Directory for Visualised Image
save_augmented_annotations_dir = os.path.join(dataset_dir, "../converted_annotations"+ "_" + dataset_basename)
save_visualised_annotations_dir = os.path.join(dataset_dir, "../visualised_annotations"+ "_" + dataset_basename)

if ENABLE_BBOX_AUGMENTATION:
    if save_augmented_annotations_dir is not None:
        if not os.path.exists(save_augmented_annotations_dir) :
            os.mkdir(save_augmented_annotations_dir)


if save_visualised_annotations_dir is not None:
    if not os.path.exists(save_visualised_annotations_dir) :
        os.mkdir(save_visualised_annotations_dir)



#Clean DB:
annotation_filenames = remove_bad_samples(dataset_dir, annotation_filenames)
testset_list = get_testset(len(annotation_filenames), testset_percentage)

print(f"Clean annotation list: {len(annotation_filenames)}")
print(f"Testset: {len(testset_list)}")



# Array of Dictionaries to store annotations
ImagesDictArray = []
AnnotationsDictArray = []
TestImagesDictArray = []
TestAnnotationsDictArray = []

# Progresbar
progress_bar = tqdm(total=len(annotation_filenames))

bbox_total_count = 0.0
sum_all_bbox_areas = 0.0
smallest_bbox_area = 0.0

# Convert annotations
for i, each in enumerate(annotation_filenames):

    basename = os.path.basename(each)
    basename = Path(basename).stem

    imagename = os.path.join(dataset_dir, str(basename + ".jpg")  )

    if not os.path.exists(imagename):
        print(f"Skipping {imagename} since it does not exists")
        continue    os.path.join(dataset_dir, "*.txt")

    image = None
    np_image = None
    im_width = im_height = 0


    # Check if the image can be openned 
    try:
        image = Image.open(imagename)
        np_image = np.array(image.convert('RGB')) 

        image_drawn = image.copy()

        im_width, im_height = image.width, image.height
        
        if ENABLE_VERBOSE_INFO: print(f"image.format {image.format}" )
    except Exception as e:
        print(f"Problems opening image: {imagename}\n" + str(e) )
        continue

    # Image Augmentation Tranform pipeline
    #   p=0.5 means that with a probability of 50%, the transform will flip the image horizontally, and with a probability of 50%,
    transform = albumentations.Compose([
        albumentations.RandomCrop(width=int(im_width*0.9),height=int(im_height*0.9), p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.7)
    ], albumentations.BboxParams(format='coco', min_area=(24*24), min_visibility=0.4)
    )


    # Create image dict
    imageDict = {}
    imageDict["id"] = i
    imageDict["file_name"] = os.path.basename(imagename)
    imageDict["coco_url"] = imagename      #abspath
    imageDict["width"] = im_width
    imageDict["height"] = im_height
    
    if i in testset_list:
        TestImagesDictArray.append(imageDict)
    else:
        ImagesDictArray.append(imageDict)

    #Augmentation
    Aug_BBoxes_Labels = []
    Aug_CategoriesArray = []
    
    draw = None
    if ENABLE_BBOX_VISUALISATION:
        draw = ImageDraw.Draw(image_drawn)
    with open(each, "r") as f:
        #Convert to Coco json

        Lines = f.readlines()

        for bbox_id, line in enumerate(Lines):
            bbox_total_count += 1
            annotationDict ={}   

            line = line.strip()
            yolo_ann = line.split(" ")

            class_label = "smoke"

            # Center coordinates
            class_idx, x, y, w, h = tuple(yolo_ann)

            # Convert YOLO coordinates to COCO format
            x,y,w,h = yolo_to_coco(x,y,w,h, im_width, im_height)

            if ENABLE_BBOX_VISUALISATION:            
                r_shape = [(x, y), (x + w -1, y + h - 1)] 
                draw.rectangle(r_shape, outline ="red", width=1) 
                #draw.text((x, y), id2label[class_idx], fill="white")  
            
            # List of bboxes for this image
            bbox_and_label = [x,y,w,h, eval(class_idx)]    

            Aug_BBoxes_Labels.append(bbox_and_label)

            # Bbox stats
            bbox_area = w*h
            sum_all_bbox_areas += bbox_area
            if smallest_bbox_area == 0.0:   smallest_bbox_area = bbox_area
            if bbox_area < smallest_bbox_area:   smallest_bbox_area = bbox_area


            # Add annotation
            annotationDict = make_annotation_dict([x,y,w,h], class_idx, image_id=i, bbox_id=bbox_id)
                                                  

            if i in testset_list:
                TestAnnotationsDictArray.append(annotationDict)
            else:
                AnnotationsDictArray.append(annotationDict)
        
        #################################
        # Visualise Annotations
        
        # Save the images to file
        
        if ENABLE_BBOX_AUGMENTATION:
            save_imagename = os.path.join(save_augmented_annotations_dir, str(basename + ".jpg"))
            image.save(save_imagename) 

        #Visualise Annotations
        if ENABLE_BBOX_VISUALISATION:
            save_imagename = os.path.join(save_visualised_annotations_dir, str(basename + ".jpg"))
            image_drawn.save(save_imagename)             

        ##########################################
        # Augment each bbox
        if ENABLE_BBOX_AUGMENTATION:
            if ENABLE_VERBOSE_INFO: print(f"\nAugmentations for image: {i}")

            for aug_index in range(num_augmentation_replicas):
                transformed = None
                
                try:
                    transformed = transform(image=np_image, bboxes=Aug_BBoxes_Labels)
                except Exception as err:
                    print("Problem transforming bbox: \n", str(err))
                    continue
                
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']

                transformed_image_pil = Image.fromarray(transformed_image)
                transformed_image_pil_drawn = transformed_image_pil.copy()
                transformed_image_draw = ImageDraw.Draw(transformed_image_pil_drawn)

                augmented_imagename = f"{basename}_aug_{aug_index}_.jpg"
                

                # Get augmented bboxes tupples
                for aug_bbox in transformed_bboxes:
                    bbox_total_count += 1

                    # Bbox stats
                    bbox_area = aug_bbox[2]*aug_bbox[3]
                    sum_all_bbox_areas += bbox_area

                    if smallest_bbox_area == 0.0:   smallest_bbox_area = bbox_area
                    if bbox_area < smallest_bbox_area:   smallest_bbox_area = bbox_area                

                    if aug_bbox[2]*aug_bbox[3] < (32*32):
                        print(f"## {augmented_imagename} bbox is small: {aug_bbox}, ")

                    if ENABLE_VERBOSE_INFO: print(f"\taug_bbox:  {aug_bbox}")

                    annotationDict = make_annotation_dict(list(aug_bbox[:4]), class_idx, image_id=i, bbox_id=bbox_id)
                    #if ENABLE_VERBOSE_INFO: print(annotationDict)

                    if i in testset_list:
                        TestAnnotationsDictArray.append(annotationDict)
                    else:
                        AnnotationsDictArray.append(annotationDict)
                        
                    if ENABLE_BBOX_VISUALISATION:
                        draw_box(transformed_image_draw, list(aug_bbox[:4]), class_idx )


                #Save Augmented images
                augmented_imagename = f"{basename}_aug_{aug_index}_.jpg"
                
                if ENABLE_BBOX_AUGMENTATION:
                    # Save the images to file
                    save_imagename = os.path.join(save_augmented_annotations_dir, augmented_imagename)
                    transformed_image_pil.save(save_imagename)

                #Visualise Annotationsfilename>0000000000</filename><size><width>1280</width><height>720</height><depth>3</depth></si
                if ENABLE_BBOX_VISUALISATION:
                    # Save the images to file
                    save_imagename = os.path.join(save_visualised_annotations_dir, augmented_imagename)
                    transformed_image_pil_drawn.save(save_imagename)
                    


        progress_bar.update(1)
    

        sleep(0.01)

###############################################
# Save Dictionaries Arrays as final Json

FinalAnnotationDict = {
    "images" : ImagesDictArray,
    "annotations": AnnotationsDictArray
}

json_data = None
try:
    json_filename = os.path.join(save_visualised_annotations_dir, "../coco_annotations_train_"+dataset_basename +".json")
    json_data_str = json.dumps(FinalAnnotationDict)
except Exception as e:
    print("Unable to dump annotations to json\n", str(e))
    sys.exit()

try:
    with open(json_filename, "w") as jsonfile:
        jsonfile.write(json_data_str)
except Exception as e:
    print(f"Unable to save the json: {json_filename}\n", str(e))
    sys.exit()


# Save Test Dictionariy Arrays as final 
CategoriesDictArray = []
categoriesDict = {
    "supercategory": "smoke",
    "id": 0,
    "name": "smoke"
}

CategoriesDictArray.append(categoriesDict)

FinalTestAnnotationDict = {
    "images" : TestImagesDictArray,
    "annotations": TestAnnotationsDictArray,
    "categories": CategoriesDictArray
}

json_data = None
try:
    json_filename = os.path.join(save_visualised_annotations_dir, "../coco_annotations_test_"+dataset_basename +".json")
    json_data_str = json.dumps(FinalTestAnnotationDict)
except Exception as e:
    print("Unable to dump annotations to json", str(e))
    sys.exit()

try:
    with open(json_filename, "w") as jsonfile:
        jsonfile.write(json_data_str)
except Exception as e:
    print(f"Unable to save the json: {json_filename}", str(e))
    sys.exit()


print(f"{bbox_total_count} Bbox Area Stats: \
      \n\tsmallest_bbox:{smallest_bbox_area}. \
      \n\taverage bbox:{sum_all_bbox_areas/bbox_total_count}")