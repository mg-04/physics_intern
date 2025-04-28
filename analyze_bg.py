from ming import *
def main():
    if len(sys.argv) != 1:
            sys.exit("Format: python average_bg.py (no additonal parameter required now!)")
    # Usage: python <program path> <.txt file path>
    folder_path = input("File folder path: ")
    
    if not os.path.exists(folder_path):
        raise Exception(f"analyze_bg.py: The specified folder path '{folder_path}' does not exist.")
    decibel = int(input("Background decibel? "))

    # file name format: <name> + "gain" + <x>"db", want to extract <name> and <x>
    path1 = folder_path + '\\background.txt'
    

    wave = read(path1)
    wave = increase_gain(wave, decibel, 11)
    #background = read("background.txt")
    #wave = subtract(wave, background)

    peaks = nfind_peak(wave, 0.01)
    zeroes = find_zeroes(wave)
    maximum = max(wave)
    print(maximum)

    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude (%)")
    #plt.ylim([-100, 100])

    plt.plot(wave[:4000], label='Waveform')

    target=0
    for i in range(20):
        target = search(target+20, "right", "zero", peaks, zeroes)
        print("Target=",target, wave[target])

        plot_point(wave, target, "", (-1)** i * maximum/15 * (i%10))
    
    plt.legend()
    plt.show()
    print("decibel =", decibel)

main()