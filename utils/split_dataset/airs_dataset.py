import os
import numpy as np 
import shutil


# divivd dataset (without annotations)
img_dir = 'data/airs/'

save_dir = 'data/Aircraft/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir_train = os.path.join(save_dir, 'train')
if not os.path.exists(save_dir_train):
    os.mkdir(save_dir_train)
save_dir_test = os.path.join(save_dir, 'test')
if not os.path.exists(save_dir_test):
    os.mkdir(save_dir_test)

# generate train dataset
f = open(os.path.join(img_dir, "images_variant_trainval.txt"))
foo = f.readlines()

for i in range(len(foo)):
    index = foo[i].find(" ")
    image_name   = foo[i][:index] + ".jpg"
    classes      = foo[i][index+1:][:-1]
    if classes.find("/")>=0:
        classes = classes[:classes.find("/")] + "_" + classes[classes.find("/")+1:]
    else:
        pass
    if classes.find(" ")>=0:
        classes = classes[:classes.find(" ")] + "_" + classes[classes.find(" ")+1:]
    else:
        pass
    # make class dir
    try:
        os.mkdir(os.path.join(save_dir_train, classes))
    except:
        print("file already exists")
    src_path = os.path.join(img_dir, 'images', image_name)
    dst_path = os.path.join(save_dir_train, classes, image_name)
    try:
        shutil.copyfile(src_path, dst_path)
        print("src:", src_path, "dst:", dst_path)
    except:
        print("error",i,foo[i])
        break

# generate test dataset
f = open(os.path.join(img_dir, "images_variant_test.txt"))
foo = f.readlines()

for i in range(len(foo)):
    index = foo[i].find(" ")
    image_name   = foo[i][:index] + ".jpg"
    classes      = foo[i][index+1:][:-1]
    if classes.find("/")>=0:
        classes = classes[:classes.find("/")] + "_" + classes[classes.find("/")+1:]
    else:
        pass
    if classes.find(" ")>=0:
        classes = classes[:classes.find(" ")] + "_" + classes[classes.find(" ")+1:]
    else:
        pass
    # make class dir
    try:
        os.mkdir(os.path.join(save_dir_test, classes))
    except:
        print("file already exists")
    src_path = os.path.join(img_dir, 'images', image_name)
    dst_path = os.path.join(save_dir_test, classes, image_name)
    try:
        shutil.copyfile(src_path, dst_path)
        print("src:", src_path, "dst:", dst_path)
    except:
        print("error",i,foo[i])
        break