import os
import fnmatch
import random
import shutil
import pandas as pd
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000     # Avoid decompression bomb warning
TARGET_FEATURE = 'carb.auc.delta'       # Column that features are derived from


# Creates augmented images to supplement training data 30% chance of generating each
def aug_image(image_in, export_name):
    # 90 rotation
    if random.randint(1, 3) == 1:
        out = image_in.rotate(90)
        out.save(export_name[:-4] + '-90' + export_name[-4:])

    # 180 rotation
    if random.randint(1, 3) == 1:
        out = image_in.rotate(180)
        out.save(export_name[:-4] + '-180' + export_name[-4:])

    # 270 rotation
    if random.randint(1, 3) == 1:
        out = image_in.rotate(270)
        out.save(export_name[:-4] + '-270' + export_name[-4:])

    # horizontal flip
    if random.randint(1, 3) == 1:
        out = image_in.transpose(Image.FLIP_LEFT_RIGHT)
        out.save(export_name[:-4] + '-h' + export_name[-4:])

    # vertical flip
    if random.randint(1, 3) == 1:
        out = image_in.transpose(Image.FLIP_TOP_BOTTOM)
        out.save(export_name[:-4] + '-v' + export_name[-4:])

    # transpose
    if random.randint(1, 3) == 1:
        out = image_in.transpose(Image.TRANSPOSE)
        out.save(export_name[:-4] + '-t' + export_name[-4:])


# Produces a squared image, resized to appropriate size
def square_image(image_in):
    width, height = image_in.size
    min_dim = min(width, height)
    image_out = image_in.crop((0, 0, min_dim, min_dim))
    image_out = image_out.resize((224, 224))    # 224x224
    return image_out


# splits strains into 2 bins based .75 quantile of target feature
def bin_median(df_in):
    df_total = pd.read_excel("Perron_phenotype-GSU-training.xlsx",
                       sheet_name="Total Database")
    df_in = df_in[['strain', TARGET_FEATURE]].copy()
    q = df_total.quantile([.75])   # get quantile ranges

    med = q.iloc[0][TARGET_FEATURE]         # quantile
    conditions = [
        (df_in[TARGET_FEATURE] < med),
        (df_in[TARGET_FEATURE] >= med)]
    choices = [0, 1]                  # 0 if below, 1 if equal or above
    df_in[TARGET_FEATURE] = np.select(conditions, choices, default=0)
    return df_in


# Splits data into training and test (90/10 split)
def split_data():
    count = 1
    index = 1
    cwd = os.getcwd()
    folder = cwd + "\images"
    for file in fnmatch.filter(os.listdir(folder), '*.jpg'):
        count += 1
        if (file[len(file) - 5] == "1"):
            index = random.randint(0, 10)
        if (index < 9):
            shutil.move(folder + '\\' + file, folder + '\\train\\' + file)

        else:
            shutil.move(folder + '\\' + file, folder + '\\test\\' + file)


########################################################################################################################

# Isolate test and training sets before resizing and augmentation
split_data()

# image dataset should be in cwd under folder named 'images'
cwd = os.getcwd()

# process each train image, augment
folder = cwd + "\images\\train"
for filename in fnmatch.filter(os.listdir(folder), '*.jpg'):
    file = folder + '\\' + filename
    image = Image.open(file)
    export_name = folder + '\\' + filename
    image = square_image(image)
    image.save(export_name)
    aug_image(image, export_name)

# process each test image
folder = cwd + "\images\\test"
for filename in fnmatch.filter(os.listdir(folder), '*.jpg'):
    file = folder + '\\' + filename
    image = Image.open(file)
    export_name = folder + '\\' + filename
    image = square_image(image)
    image.save(export_name)

# process each validation image
folder = cwd + "\images\\validation"
for filename in fnmatch.filter(os.listdir(folder), '*.jpg'):
    file = folder + '\\' + filename
    image = Image.open(file)
    export_name = folder + '\\' + filename
    image = square_image(image)
    image.save(export_name)

# Create csv with strain numbers and labels (based on median)
df = pd.read_excel("Perron_phenotype-GSU-training.xlsx",
                   sheet_name="Isolates w Microscopy")
df = bin_median(df)
df.to_csv('target_labels.csv', encoding='utf-8', index=False)
