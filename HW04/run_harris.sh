#!/bin/bash
harris=/Users/wang3450/Desktop/ECE661/HW04/harris.py
book="book"
fountain="fountain"
checkerboard="checkerboard"
locker="locker"
NCC="NCC"
SSD="SSD"

# Book Image Set:
# Sigma = 5
python3 $harris $book 5 $NCC
python3 $harris $book 5 $SSD

# Book Image Set:
# Sigma = 10
python3 $harris $book 10 $NCC
python3 $harris $book 10 $SSD


# Fountain Image Set:
# Sigma = 1.2
python3 $harris $fountain 1.2 $NCC
python3 $harris $fountain 1.2 $SSD

# Fountain Image Set:
# Sigma = 5
python3 $harris $fountain 5 $NCC
python3 $harris $fountain 5 $SSD

# Fountain Image Set:
# Sigma = 10
python3 $harris $fountain 10 $NCC
python3 $harris $fountain 10 $SSD

# Checkerboard Image Set:
# Sigma = 1.2
python3 $harris $checkerboard 1.2 $NCC
python3 $harris $checkerboard 1.2 $SSD

# Checkerboard Image Set:
# Sigma = 5
python3 $harris $checkerboard 5 $NCC
python3 $harris $checkerboard 5 $SSD

# Checkerboard Image Set:
# Sigma = 10
python3 $harris $checkerboard 10 $NCC
python3 $harris $checkerboard 10 $SSD

# Locker Image Set:
# Sigma = 1.2
python3 $harris $locker 1.2 $NCC
python3 $harris $locker 1.2 $SSD

# Locker Image Set:
# Sigma = 5
python3 $harris $locker 5 $NCC
python3 $harris $locker 5 $SSD

# Locker Image Set:
# Sigma = 10
python3 $harris $locker 10 $NCC
python3 $harris $locker 10 $SSD
