import os
import numpy as np 
import shutil


# divivd dataset (without annotations)
img_dir = 'data/birds/'

save_dir = 'data/Birds/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir_train = os.path.join(save_dir, 'train')
if not os.path.exists(save_dir_train):
    os.mkdir(save_dir_train)
save_dir_test = os.path.join(save_dir, 'test')
if not os.path.exists(save_dir_test):
    os.mkdir(save_dir_test)

f2 = open(os.path.join(img_dir, "images.txt"))
foo = f2.readlines()

f = open(os.path.join(img_dir, "train_test_split.txt"))
bar = f.readlines()

f3 = open(os.path.join(img_dir, "image_class_labels.txt"))
baz = f3.readlines()

for i in range(len(foo)):
    image_id   = foo[i].split(" ")[0]
    image_path = foo[i].split(" ")[1][:-1]
    image_name = image_path.split("/")[1]
    is_train = int(bar[i].split(" ")[1][:-1])
    classes = baz[i].split(" ")[1][:-1].zfill(2)
    # split train & test data
    if is_train:
        # make class dir
        try:
            os.mkdir(os.path.join(save_dir_train, classes))
        except:
            print("file already exists")
        src_path = os.path.join(img_dir, 'images', image_path)
        dst_path = os.path.join(save_dir_train, classes, image_name)        
    else:
        # make class dir
        try:
            os.mkdir(os.path.join(save_dir_test, classes))
        except:
            print("file already exists")
        src_path = os.path.join(img_dir, 'images', image_path)
        dst_path = os.path.join(save_dir_test, classes, image_name)        
    shutil.copyfile(src_path, dst_path)
    print("src:", src_path, "dst:", dst_path)