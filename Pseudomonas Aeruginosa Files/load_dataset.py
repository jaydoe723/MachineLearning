from random import shuffle
import glob
import sys
import re
from PIL import Image
import tensorflow as tf
import pandas as pd

TARGET_FEATURE = 'carb.auc.delta'
target_df = pd.read_csv("target_labels.csv")


# Convert to form the tf will understand
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Convert to form the tf will understand
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Find strain number then then find corresponding label
def find_label(filename):
    regex = re.search("PIL-\d*", filename)
    strain = filename[regex.start()+4: regex.end()]
    strain = int(strain)
    df = target_df.loc[target_df['strain'] == strain]
    if df.empty:
        label = 0
    else:
        label = df.iloc[0][TARGET_FEATURE]
    return label


def create_record(out_filename, addrs):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    print("Creating: " + out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 5 images
        if not i % 5:
            print('{}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        # Load the image
        filename = addrs[i]
        img = Image.open(filename)

        if img is None:
            continue

        # Generate label
        label = find_label(filename)

        # Create a feature
        feature = {
            'image_raw': _bytes_feature(img.tobytes()),
            'label': _int64_feature(label)
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    print("Done")
    writer.close()
    sys.stdout.flush()


train_path = 'images/train/*.jpg'
test_path = 'images/test/*.jpg'
val_path = 'images/validation/*.jpg'
# read addresses and labels from the train folder
train_addrs = glob.glob(train_path)
test_addrs = glob.glob(test_path)
val_addrs = glob.glob(val_path)

# shuffle data
shuffle(train_addrs)
shuffle(test_addrs)
shuffle(val_addrs)


# create records files
create_record('train.tfrecords', train_addrs)
create_record('test.tfrecords', test_addrs)
create_record('val.tfrecords', val_addrs)
