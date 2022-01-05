import numpy as np
from PIL import Image
import os
import csv
import pdb
from tqdm import tqdm
import argparse

## -- size option
parser = argparse.ArgumentParser()
parser.add_argument\
('-s', '--size', type = int, default = 128, help = "image size, ex) 128 x 128")

args = parser.parse_args()
    
## -- search images file_path
csv_file = "by_merge.csv"
with open(csv_file, newline= "") as f:
    reader = csv.reader(f)
    tmp = list(reader)
    column_list = tmp[0]
    total_list = tmp[1:]
pdb.set_trace()


## -- convert png to numpy and save at new dir
for row in tqdm(total_list):
    png_path = os.path.join("./Font/Font",row[0])
    image = Image.open(png_path)

    # resize (original size = 128 x 128)
    if args.size != 128:
        image = image.resize((args.size, args.size), Image.BICUBIC)    
        
    # normalize 0 - 1
    pixel = np.array(image)
    pixel = pixel.astype("float32")
    pixel /= 255.0

    # label 
    label = int(row[1])
    label = np.int64(label)
    combined = (pixel, label)

    # np.save
    save_path = png_path.replace("./Font/Font/","./Font_npy_"+str(args.size)+"/")    
    save_path = save_path.replace(".png", ".npy")
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.save(save_path, combined, allow_pickle = True)

