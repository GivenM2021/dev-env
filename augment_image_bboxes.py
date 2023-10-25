from PIL import Image
import numpy as np
import os, pathlib
import albumentations as Alb

user_home = None
output_dir = None

    
try:
    user_home = os.getenv("HOME")
    if output_dir is None:
        output_dir = os.path.join(user_home, "Pictures/image_augmentation_example")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

except Exception:
    print("Problem creating the output directory")

image_filename = "/home/ipsaas/Pictures/cat.jpg"
image_basename = os.path.basename(image_filename)

image = Image.open("/home/ipsaas/Pictures/cat.jpg")
np_image = np.array(image)

im_width, im_height = image.width, image.height

# Tranform pipeline
image_transform = Alb.Compose([
    Alb.RandomCrop(width=int(im_width*0.8),height=int(im_height*0.8)),
    Alb.HorizontalFlip(p=0.5),
    Alb.RandomBrightnessContrast(p=0.2)
])

"""
# Read json file to use Albumentations Image dTransforms
for each image:
    find all annotations that have similar 'image_id
    save all the annotations in a 'bboxes_array list
    create another array list of  'category_ids' that corregesponds to 'bboxes_array'

"""



# for i, each_image in enumerate(augmentations):
for i in range(10):

    augmentations = image_transform(image = np_image)
    
    transformed_image = augmentations["image"]

    pil_image = Image.fromarray(transformed_image)

    image_base_split = image_basename.split(".")

    image_filename = image_base_split[0] + "_"+ str(i) + "." +image_base_split[1]

    image_filename = os.path.join(output_dir, image_filename)

    print(f"image_file {image_filename}")

    pil_image.save(image_filename)



