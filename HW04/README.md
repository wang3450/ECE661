# Homework 4
This directory consists of all the necessary files to complete homework 4. The assignment details are listed in HW3-Fall2022.pdf

## Harris Corner Detector
Execute the following:
```sh
python3 harris.py <imageSet> <sigma>
```
imageSet (str): which set to load {book, fountain, checkerboard, locker}

sigma (float): how much smoothing

## Scale Invariant Feature Transform
Execute the following:
```sh
python3 sift.py <imageSet>
```
imageSet (str): which set to load {book, fountain, checkerboard, locker}


## SuperGlue/SuperPoint
Create conda environment
```sh
conda create --name superglue python=3.6
conda activate superglue
pip install -r SuperGluePretrainedNetwork/requirements.txt
```


Execute the following:
```sh
#!/bin/bash
img1=./ece661_sample_images/1.jpg
img2=./ece661_sample_images/2.jpg
img3=./ece661_sample_images/3.jpg
img4=./ece661_sample_images/4.jpg
img5=./ece661_sample_images/5.jpg
outDir=./output_images

book_1=/Users/wang3450/Desktop/ECE661/HW04/input_images/books_1.jpeg
book_2=/Users/wang3450/Desktop/ECE661/HW04/input_images/books_2.jpeg
checkerboard_1=/Users/wang3450/Desktop/ECE661/HW04/input_images/checkerboard_1.jpg
checkerboard_2=/Users/wang3450/Desktop/ECE661/HW04/input_images/checkerboard_2.jpg
locker_1=/Users/wang3450/Desktop/ECE661/HW04/input_images/locker_1.jpg
locker_2=/Users/wang3450/Desktop/ECE661/HW04/input_images/locker_2.jpg
fountain_1=/Users/wang3450/Desktop/ECE661/HW04/input_images/fountain_1.jpg
fountain_2=/Users/wang3450/Desktop/ECE661/HW04/input_images/fountain_2.jpg


python3 superglue_ece661.py $img1 $img2 $outDir
python3 superglue_ece661.py $img1 $img3 $outDir
python3 superglue_ece661.py $img1 $img4 $outDir
python3 superglue_ece661.py $img1 $img5 $outDir
python3 superglue_ece661.py $img2 $img3 $outDir
python3 superglue_ece661.py $img2 $img4 $outDir
python3 superglue_ece661.py $img2 $img5 $outDir
python3 superglue_ece661.py $img3 $img4 $outDir
python3 superglue_ece661.py $img3 $img5 $outDir
python3 superglue_ece661.py $img4 $img5 $outDir

python3 superglue_ece661.py $book_1 $book_2 $outDir
python3 superglue_ece661.py $fountain_1 $fountain_2 $outDir
python3 superglue_ece661.py $locker_1 $locker_2 $outDir 
python3 superglue_ece661.py $checkerboard_1 $checkerboard_2 $outDir
```

## Mathematical Theory and Explanation
hw4_joseph_wang.pdf contains detailed step by step explanations of the necessary steps to solve the various tasks required.