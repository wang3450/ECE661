#!/bin/bash
harris=/Users/wang3450/Desktop/ECE661/HW04/harris.py
book="book"
fountain="fountain"
checkerboard="checkerboard"
locker="locker"
NCC="NCC"
SSD="SSD"

# Book Image Set:
# Sigma = 1.6, 2
python3 $harris $book 1.6 $NCC
python3 $harris $book 1.6 $SSD
python3 $harris $book   2 $NCC
python3 $harris $book   2 $SSD

# Fountain Image Set:
# Sigma = 1.6, 2
python3 $harris $fountain 1.6 $NCC
python3 $harris $fountain 1.6 $SSD
python3 $harris $fountain   2 $NCC
python3 $harris $fountain   2 $SSD

# Checkerboard Image Set:
# Sigma = 1.6, 2
python3 $harris $checkerboard 1.6 $NCC
python3 $harris $checkerboard 1.6 $SSD
python3 $harris $checkerboard   2 $NCC
python3 $harris $checkerboard   2 $SSD

# Locker Image Set:
# Sigma = 1.6, 2
python3 $harris $locker 1.6 $NCC
python3 $harris $locker 1.6 $SSD
python3 $harris $locker   2 $NCC
python3 $harris $locker   2 $SSD



