# takes an csv file specifying file name and time bounds, and outputs a csv file containing the integrals
# Input: File folder, background gain
#   Reads a .csv file named "integral_time.csv" that specifies the file name and time interval of the integral
#    .csv row format:
#       spot name (no gain needed)          t1          t2          t3
# Output: a .csv file, with the two integrated values: i1 and i2 appenended after each row
#   (i1 is the integral between t1 and t2, i2 is between t2 and t3)

from ming import *

def main():
    if len(sys.argv) != 1:
            sys.exit("Format: python auto_integrate.py (no additonal parameter required now!)")
    while(1):
        data = []   # List for the final .csv output
        folder_path = input("File folder path: ")
        if(folder_path == 'quit'):
            break

        # locates and reads the background file
        background_gain = int(input("Background gain: (default=11): "))
        background = read(folder_path + "\\background.txt")
        background = increase_gain(background, background_gain, 11)

        # locates and stores the integral intsruction file
        csv_file_path = folder_path + "\\integral_time.csv"
        integral_times = []
        # Open the CSV file and read its content
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                integral_times.append(row)

        for row in integral_times:
            data_row = row  # The program appends integrated values after each row
            ffile_name = row[0] # what might be the file name
            t1 = int(row[1])
            t2 = int(row[2])
            t3 = int(row[3])

            # Guess the file name (gain) and read whatever that does not produce an error
            file_extensions = ['', ' gain 0db', ' gain 5db', ' gain 11db', ' gain 16db', ' gain 21db']
            for file_extension in file_extensions:
                try:
                     print("Trying", folder_path + '\\' + ffile_name + file_extension + '.txt')
                     wave = read(folder_path + '\\' + ffile_name + file_extension + '.txt')
                     break
                except FileNotFoundError:
                     continue
            else:   # executes after the for loop completes without break, aka no file name found
                 raise FileNotFoundError("Auto_integrate.py: File does not exist, or uses an unknown gain!")
            
            # Reads the wave decibel from the gain. If not, assume a default value
            try:
                print(file_extension.split()[1])
                decibel = int(file_extension.split()[1][:-2])
            except:     # default gain: 11 for 1 MHz, 16 for 2.25
                if ("2.25 MHz" in folder_path or "2.25MHz" in folder_path):
                    decibel = 16
                elif ("1 MHz" in folder_path or "1MHz" in folder_path):
                    decibel = 11
                else:
                    raise Exception("Auto_integrate.py: Unknown gain for this frequency!")
            print("Wave db =", decibel, "Bg db =", background_gain)

            # Adjust the wave
            wave = increase_gain(wave, decibel, 11)
            wave = subtract(wave, background)

            # Set up the plot
            plt.xlabel("Time (us)")
            plt.ylabel("Amplitude (%)")
            #plt.ylim([-100, 100])
            plt.plot(wave[:4000], label='Waveform')
            
            # Integrate
            i1 = integrate(wave, t1, t2, "rms")
            print(i1)
            plot_integral(wave, t1, t2, i1)
            i2 = integrate(wave, t2, t3, "rms")
            print(i2)
            plot_integral(wave, t2, t3, i2)

            # Save the plot and data
            plt.savefig(folder_path+'/'+ffile_name)
            plt.clf()   # Clear to avoid overlap
            data_row.extend([i1, i2])
            data.append(data_row)

        # Write to "integrals.csv"
        csv_file_path = folder_path+'/'+'integrals.csv'
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(data)
        print(f'Integral written to {csv_file_path}')

main()