import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from ming import *

# Supported format: bg <gain><file#>.txt
# E. bg 211.txt

def main():
    if len(sys.argv) != 1:
            sys.exit("Format: python avg_bg.py (no additonal parameter required now!)")
    folder_path = input("File folder path: ")
    gain = input("Background gain: ")   # not converted to int yet

    if not os.path.exists(folder_path):
        raise Exception("avg_bg.py: Folder path does not exist!")

    bg_number = int(input("Number of bg files:"))
    sum = []
    min_len = 99999
    for i in range(1, bg_number+1):
        path = folder_path + '\\bg '+gain+str(i)+'.txt'
        wave = read(path)
        min_len = min(len(wave), min_len)
    for i in range(1, bg_number + 1):
        path = folder_path + '\\bg '+gain+str(i)+'.txt'
        wave = read(path)[:min_len - 1]
        if (sum == []):
            sum = wave
        else:
            for i in range(len(sum)):
                sum[i] += wave[i]

    sun = []
    for pt in sum:
        sun.append(pt/bg_number)
    write(folder_path+"\\background.txt", sun)
    print("File written in \"background.txt\"")

   
main()

