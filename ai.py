from ming import *
def main():
    if len(sys.argv) != 1:
            sys.exit("Format: python ai.py (no additonal parameter required now!)")
    while(1):
        data = []
        folder_path = input("File folder path: ")
        if(folder_path == 'quit'):
            break
        background_gain = int(input("Background gain: (default=11): "))
        background = read(folder_path + "\\background.txt")
        background = increase_gain(background, background_gain, 11)
        while(1):
            data_row = []
            ffile_name = input("File name: ")   # needn't specify the gain here
            if(ffile_name == 'quit'):
                break 

            # Guess the file name (gain) and read whatever that does not produce an error
            file_extensions = ['', ' gain 0db', ' gain 5db', ' gain 11db', ' gain 16db', ' gain 21db']
            for file_extension in file_extensions:
                try:
                     file_name = folder_path + '\\' + ffile_name + file_extension + '.txt'
                     print("Trying", file_name)
                     wave = read(file_name)
                     break
                except FileNotFoundError:
                     continue
            else:   # executes after the for loop completes without break, aka no file name found
                 print("Auto_integrate.py: File does not exist, or uses an unknown gain!")
                 continue
            
            # Reads the wave decibel from the gain. If not, assume a default value
            try:
                print(file_extension.split()[1])
                decibel = int(file_extension.split()[1][:-2])
            except:     # default gain: 11 for 1 MHz, 16 for 2.25
                if ("2.25 MHz" in folder_path):
                    decibel = 16
                elif ("1 MHz" in folder_path):
                    decibel = 11
                else:
                    raise Exception("Auto_integrate.py: Unknown gain for this frequency!")
            print("Wave db =", decibel, "Bg db =", background_gain)

            data_row.append(file_name)
            
            wave = increase_gain(wave, decibel, 11)
            wave = subtract(wave, background)
            peaks = nfind_peak(wave, 2)
            zeroes = find_zeroes(wave)
            plt.xlabel("Time (us)")
            plt.ylabel("Amplitude (%)")
            #plt.ylim([-100, 100])
            plt.plot(wave[:4000], label='Waveform')
            target=0
            for i in range(100):
                target = search(target+20, "right", "zero", peaks, zeroes)
                print("Target=",target, wave[target])
                plot_point(wave, target, "", (-1)** i * 1 * (i%10))
            plt.legend()
            plt.show()
            print("decibel =", decibel)

            while(1):
                t1 = int(input("t1: "))
                t2 = int(input("t2: "))
                t3 = int(input("t3: "))

                
                plt.xlabel("Time (us)")
                plt.ylabel("Amplitude (%)")
                #plt.ylim([-100, 100])

                plt.plot(wave[:4000], label='Waveform')
                
                i1 = integrate(wave, t1, t2, "rms")
                print(i1)
                plot_integral(wave, t1, t2, i1)
                i2 = integrate(wave, t2, t3, "rms")
                print(i2)
                plot_integral(wave, t2, t3, i2)

                plt.legend()
                plt.savefig(folder_path+'/'+file_name.split()[0])
                plt.show()
                confirm_integral = input("Comfirm? [Y]/N ")
                if (confirm_integral == ''):
                    break

            data_row.extend([t1, t2, t3, i1, i2])
            data.append(data_row)


        # Specify the file path for the CSV file
        csv_file_path = folder_path+'/'+'integrals.csv'

        # Open the CSV file in write mode
        with open(csv_file_path, 'w', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)

            # Write the data to the CSV file
            csv_writer.writerows(data)

        print(f'Data has been written to {csv_file_path}')

main()