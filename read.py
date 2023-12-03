# TO DO:    automate the algorithm
#           use energy (square)
#           record gain/attenuation constant, and number of cycles


#python "C:\Users\13764\Documents\academics\sr\fall 2023\physics intern\matec\read.py" "C:\Users\13764\Documents\academics\sr\fall 2023\physics intern\2022-2023 Ultrasonics\3-22-17 top (ds)\top\7-7-23 Z+ gel heat\heat 2\200.txt"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from ming import *

def main():
    
    if len(sys.argv) != 2:
            sys.exit("Usage: python read.py")
    # Usage: python <program path> <.txt file path>
    path1 = sys.argv[1]+'.txt'

    with open(path1) as f:
        lines = f.readlines()

    result1 = extract_data(lines)
   
    write(path1, result1[0])
    
   
main()

