# TA_EEE4178_Project_2021_preparation
For Poor little next TA

## 0. Original Data preparation
* I downloaded dataset from kaggle, but cannot find it anymore.
* Original dataset comprises two file => Font directory containing font images and train.csv
```bash
├── Font
   ├── Font
   ├── Sample008
   └── Sample003
             '
             '  
             '
├── train.csv
``` 
* Label : not font, character (a/ b/ c/ ,,, similar to EMNIST) 

## 1. By_Merge.py
* delete one of uppercase letter or lower case letter of characters that are difficult to distinguish between uppercase and lowercase letters. 
* delete_list = ["c", "k", "l", "O", "p", "s", "v", "w", "x", "z"]
* After deletion, number of total class would be 62 -> 52 (10 reduced)
* After run the python code, 'by_merge.csv' file will be located in root directory.

## 2. Png2npy.py
* convert png -> npy
* Resize ?x? -> 90 x 90
* Normalize -> 0-1 scaled
* After run the python code, 'Font_npy_90' file will be located in the root directory.
* Preprocessed Dataset can be downloaded from TA_EEE4178_Project_2021
~~~
python png2npy.py --90
~~~

## 3. Split train & Test & Valid data
* make each set of data disjoint per each class.
* After run the python code, 'Font_npy_90_train','Font_npy_90_train','Font_npy_90_test' will be generated. 
~~~
split_train_val_test.py --90
~~~


