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
