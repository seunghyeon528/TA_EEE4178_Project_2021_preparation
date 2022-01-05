import numpy as np
import pandas as pd
import cv2
import os
import pdb
import csv

## -- read csv
csv_file = "./train.csv"
with open(csv_file, newline= "") as f:
    reader = csv.reader(f)
    tmp = list(reader)
    column_list = tmp[0]
    total_list = tmp[1:]

pdb.set_trace()


## -- count number of data per each character
char_num_dict = {}
for row in total_list:
    if row[1].strip() not in char_num_dict.keys():
        char_num_dict[str(row[1]).strip()] = 0
    char_num_dict[str(row[1]).strip()] += 1

pdb.set_trace()


## -- delete confusing character
delete_list = ["c", "k", "l", "O", "p", "s", "v", "w", "x", "z"]
selected_list = [x for x in total_list if x[1].strip() not in delete_list]
pdb.set_trace()


# -- count number of data per each character
char_num_dict = {}
for row in selected_list:
    if row[1].strip() not in char_num_dict.keys():
        char_num_dict[str(row[1]).strip()] = 0
    char_num_dict[str(row[1]).strip()] += 1

assert len(char_num_dict.keys()) == 52
pdb.set_trace()


## -- encoding each character
all_fonts = [x[1] for x in selected_list]
unique_fonts = set(all_fonts) # take unique values
unique_fonts = list(unique_fonts)
unique_fonts.sort()
pdb.set_trace()
encoding_numbers = [x for x in range(52)]
encoding_items = zip(unique_fonts, encoding_numbers)
encoding_dict = {k:v for (k,v) in encoding_items}
pdb.set_trace()


## -- rewrite csv
selected_list = [[x[0].replace("../input/english-fontnumber-recognition/Font/Font/",""),encoding_dict[x[1]]] for x in selected_list]
csv_path = "./by_merge.csv"
f = open(csv_path, "w", encoding = "utf-8",newline = "")
wr = csv.writer(f)
wr.writerow(column_list)
for row in selected_list:
    wr.writerow(row)
f.close()