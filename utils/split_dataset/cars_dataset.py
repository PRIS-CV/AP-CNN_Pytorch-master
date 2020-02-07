from scipy.io import loadmat
import cv2
import os


# divivd dataset (without annotations)
img_dir = 'data/cars/'

save_dir = 'data/StandCars/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_dir_train = os.path.join(save_dir, 'train')
if not os.path.exists(save_dir_train):
    os.mkdir(save_dir_train)
save_dir_test = os.path.join(save_dir, 'test')
if not os.path.exists(save_dir_test):
    os.mkdir(save_dir_test)

m = loadmat(os.path.join(img_dir, "cars_annos.mat"))
info = m['annotations'][0]
for img_info in info:
    img_name = img_info[0][0]
    img_path = os.path.join(img_dir, img_name)
    classes = str(int(img_info[-2]))
    # make class dir
    try:
        os.mkdir(os.path.join(save_dir_train, classes))
    except:
        print("file already exists")
    try:
        os.mkdir(os.path.join(save_dir_test, classes))
    except:
        print("file already exists")
    
    # split to train/test     
    img_test_flag = int(img_info[-1])
    if img_test_flag:
        save_path = os.path.join(save_dir_test, classes, img_name[8:])
        img = cv2.imread(img_path)
        # save origin image
        cv2.imwrite(save_path, img)
    else:
        save_path = os.path.join(save_dir_train, classes, img_name[8:])
        img = cv2.imread(img_path)
        # save origin image
        cv2.imwrite(save_path, img)