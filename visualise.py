from tqdm import tqdm
import albumentations as A

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

import os, sys

import glob
from time import sleep

from pathlib import Path



abs_dataset_path = "/home/ipsaas/data_ml/datasets/custom_data"
dataset_basename = os.path.basename(abs_dataset_path)
dataset_basename = Path(dataset_basename).stem

dataset_dir = abs_dataset_path #os.path.join( os.curdir, "annotations" )

print(dataset_dir)

annotation_filenames = glob.glob(os.path.join(dataset_dir, "*.txt") )

print(annotation_filenames)

# Save Directory for Visualised Image
save_visualised_annotations_dir = os.path.join(dataset_dir, "../visualised_annotations"+ "_" + dataset_basename)

if save_visualised_annotations_dir is not None:
    if not os.path.exists(save_visualised_annotations_dir) :
        os.mkdir(save_visualised_annotations_dir)


# Read the files
progress_bar = tqdm(total=len(annotation_filenames))

#for each in tqdm(annotation_filenames):
for each in annotation_filenames:

    basename = os.path.basename(each)
    basename = Path(basename).stem

    imagename = os.path.join(dataset_dir, str(basename + ".jpg")  )

    if not os.path.exists(imagename):
        print(f"Skipping {imagename} since it does not exists")
        continue

    image = None
    im_width = im_height = 0
    try:

        image = Image.open(imagename) 

        im_width, im_height = image.width, image.height
        #image.show()
    except Exception as e:
        print(f"Problems loading {imagename}\n" + str(e) )
        continue
        #sys.exit()


    draw = ImageDraw.Draw(image)
    with open(each, "r") as f:
        #Convert to Coco json

        Lines = f.readlines()

        if len(Lines) == 0:
            print(f"{basename} has empty annotations")

        for line in Lines:
            line = line.strip()

            yolo_ann = line.split(" ")

            #print(yolo_ann)

            #Draw box
            class_idx = "smoke"
            _, x, y, w, h = tuple(yolo_ann)

            x = eval(x)*im_width        #x center
            y = eval(y)*im_height       #y center
            w = eval(w)*im_width
            h = eval(h)*im_height

            #shape = [(x-5, y-5), (w-5, h-5)]          #(shape w,h) , (position x, y)
            #draw.ellipse(shape, fill ="yellow", outline ="pink") 

            # Top left corner from center point
            left_x = x - (w/2)
            left_y = y - (h/2)

            r_shape = [(left_x, left_y), (left_x + w -1, left_y + h - 1)] 

            draw.rectangle(r_shape, outline ="red", width=1) 
            #draw.point(((x, y), (left_x, left_y) ), fill=(255, 255, 0))

            #print(f"x, y, w, h = {x}, {y}, {w}, {h}")



            #draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
            #draw.text((x, y), id2label[class_idx], fill="white")        

        #Visualise Annotations
        open_cv_image = np.array(image.convert('RGB')) 
        # Convert RGB to BGR 
        #open_cv_image = open_cv_image[:, :, ::-1].copy()

        # plt.imshow(open_cv_image)
        # plt.show()

        # plt.close("all")

        #Save the images
        save_imagename = os.path.join(save_visualised_annotations_dir, str(basename + ".jpg"))
        image.save(save_imagename)


        progress_bar.update(1)
    
    sleep(0.02)




# hide image
import psutil
for proc in psutil.process_iter():
    if proc.name() == "display":
        proc.kill()