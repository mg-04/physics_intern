# TO DO:    automate the algorithm
#           use energy (square)
#           record gain/attenuation constant, and number of cycles


#python "C:\Users\13764\Documents\academics\sr\fall 2023\physics intern\matec\read.py" "C:\Users\13764\Documents\academics\sr\fall 2023\physics intern\2022-2023 Ultrasonics\3-22-17 top (ds)\top\7-7-23 Z+ gel heat\heat 2\200.txt"
from ming import *

def main():
    if len(sys.argv) != 1:
            sys.exit("Format: python auto_read.py (no additonal parameter required now!)")
    # Usage: python <program path> <.txt file path>

    while(1):
        folder_path = input("File folder path: ")

        if folder_path == "quit":
            break

        if not os.path.exists(folder_path):
            print(f"analyze_bg.py: The specified folder path '{folder_path}' does not exist.")
            continue
            #raise Exception(f"analyze_bg.py: The specified folder path '{folder_path}' does not exist.")

        
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)

                try:
                    with open(file_path) as f:
                        lines = f.readlines()

                    result1 = extract_data(lines)
                    write(file_path, result1[0])
                    print(f"Parsed: '{filename}'")

                except:
                    print(f"Error: '{filename}' cannot be parsed!")
        
main()

