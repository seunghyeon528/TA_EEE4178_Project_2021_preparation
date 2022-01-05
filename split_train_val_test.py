import json
import os
import glob
import random
import shutil
import pdb
from tqdm import tqdm
import argparse

# size
parser = argparse.ArgumentParser()
parser.add_argument\
('-s', '--size', type = int, default = 128, help = "image size, ex) 128 x 128")

args = parser.parse_args()

# hyperparameter
num_test = 150
num_val = 150
random.seed(5028)

# splitted_list initialization
test_list = []
train_list = []
val_list = []

root_dir = "./Font_npy_{}".format(str(args.size))

class_dir_list = os.listdir(root_dir)
pdb.set_trace()

for class_dir in class_dir_list:
    all_npy = glob.glob(os.path.join(root_dir, class_dir, "*.npy"))
    random.shuffle(all_npy)

    #split
    test_npy = all_npy[:num_test]
    val_npy = all_npy[num_test:num_test+num_val]
    train_npy = all_npy[num_test+num_val:]

    test_list.extend(test_npy)
    train_list.extend(train_npy)
    val_list.extend(val_npy)

pdb.set_trace()


# copy file - train_dir, val_dir, test_dir 지우고 시작해야
train_dir = root_dir + "_train"
val_dir = root_dir + "_val"
test_dir = root_dir + "_test"

for origin_path in tqdm(train_list):
    copy_path = origin_path.replace(root_dir, train_dir)
    dir_name = os.path.dirname(copy_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    shutil.copy(origin_path, copy_path)
    #pdb.set_trace()

for origin_path in tqdm(val_list):
    copy_path = origin_path.replace(root_dir, val_dir)
    dir_name = os.path.dirname(copy_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    shutil.copy(origin_path, copy_path)
    #pdb.set_trace()

for origin_path in tqdm(test_list):
    copy_path = origin_path.replace(root_dir, test_dir)
    dir_name = os.path.dirname(copy_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    shutil.copy(origin_path, copy_path)
    #pdb.set_trace()


# assert non-overlapping
assert len (set(test_list) & set(train_list)) == 0
assert len (set(val_list) & set(train_list)) == 0
assert len (set(val_list) & set(test_list)) == 0