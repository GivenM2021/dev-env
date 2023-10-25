from tqdm import tqdm

from PIL import Image, ImageDraw

import os, sys
import json
import glob
from time import sleep
from pathlib import Path

from yolo_db_functions import get_testset, remove_bad_samples, yolo_to_coco

ENABLE_BBOX_VISUALISATION = False

testset_percentage = 15

abs_dataset_path = "/home/ipsaas/data_ml/datasets/custom_data"
dataset_basename = os.path.basename(abs_dataset_path)
dataset_basename = Path(dataset_basename).stem

dataset_dir = abs_dataset_path #os.path.join( os.curdir, "annotations" )

print(dataset_dir)

annotation_filenames = glob.glob(os.path.join(dataset_dir, "*.txt") )


# Save Directory for Visualised Image
save_visualised_annotations_dir = os.path.join(dataset_dir, "../converted_annotations"+ "_" + dataset_basename)

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

# Convert annotations
for i, each in enumerate(annotation_filenames):

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
        #image.show()
    except Exception as e:
        print(f"Problems loading {imagename}\n" + str(e) )
        continue

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


    draw = None
    if ENABLE_BBOX_VISUALISATION:
        draw = ImageDraw.Draw(image)
    with open(each, "r") as f:
        #Convert to Coco json

        Lines = f.readlines()

        for bbox_id, line in enumerate(Lines):
            annotationDict ={}   

            line = line.strip()

            yolo_ann = line.split(" ")

            #print(yolo_ann)

            class_label = "smoke"

            # Center coordinates
            bbox_label, x, y, w, h = tuple(yolo_ann)

            x,y,w,h = yolo_to_coco(x,y,w,h, im_width, im_height)
            #shape = [(x-5, y-5), (w-5, h-5)]          #(shape w,h) , (position x, y)
            #draw.ellipse(shape, fill ="yellow", outline ="pink") 

            if ENABLE_BBOX_VISUALISATION:            
                r_shape = [(x, y), (x + w -1, y + h - 1)] 
                draw.rectangle(r_shape, outline ="red", width=1) 


            #draw.text((x, y), id2label[class_idx], fill="white")  

            # Add annotation
            annotationDict = {
                "iscrowd": 0,
                "id" : bbox_id,
                "image_id": i,
                "category_id": eval(bbox_label),
                "area": int(w*h),
                "bbox": [x,y,w,h]
            }

            if i in testset_list:
                TestAnnotationsDictArray.append(annotationDict)
            else:
                AnnotationsDictArray.append(annotationDict)
            


        #Visualise Annotations
        if ENABLE_BBOX_VISUALISATION:
            #open_cv_image = np.array(image.convert('RGB')) 

            # Save the images to file
            save_imagename = os.path.join(save_visualised_annotations_dir, str(basename + ".jpg"))
            image.save(save_imagename)

        


        progress_bar.update(1)
    
    sleep(0.02)

# Save Dictionariy Arrays as final Json
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


