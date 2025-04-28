# auto_analyze_integrate_v2.py
# LONGITUDINAL wave only!

from ming import *
def main(frequency):
    if len(sys.argv) != 1:
            sys.exit("Format: python ai.py")

    data = []
    # locates and stores the integral intsruction file
    csv_file_path = "C:\\Users\\13764\\Documents\\ACADEMICS\\sr\\physics intern\\matec\\Compressional_time_integral - " + frequency + " MHz.csv"
    integral_times = []
    # Open the CSV file and read its content
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            integral_times.append(row)

    cube = ""
    face = ""
    date = ""
    for row in integral_times[1:]:
        if row[3] == "":    # if row (spot number) is empty
            data.append([])
            continue
        
        # read the parameters
        if row[0] != "":
            cube = row[0]
        if row[1] != "":
            face = row[1]
        if row[2] != "":
            date = row[2]
        spot = row[3]
        t1 = int(row[4])
        try:    # pulse not found case
            t2 = int(row[5])
        except:
            continue
        t3 = int(row[6])
        data_row = [cube, face, date, spot, t1, t2, t3]

        wave = read_guess(cube, face, frequency, date, spot, plt_gain=11)
       
        # Set up the plot
        plt.xlabel("Time (us)")
        plt.ylabel("Amplitude (%)")
        #plt.ylim([-100, 100])
        if frequency == '1':
            x_lim = 4000
        elif frequency == '2.25':
            x_lim = 2500
        elif frequency == '5':
            x_lim = 2300
        elif frequency == '10':
            x_lim = 2200
        plt.plot(wave[:x_lim], label='Waveform', linewidth=0.75)
        plt.xticks(np.arange(0, x_lim, step=1000), labels=[str(i/100) for i in range(0, x_lim, 1000)])
        plt.xlim([0, x_lim])

        print(t1)
        print(t2)
        print(t3)
        pulse_width = t3-t2
        
        # Integrate
        i1 = integrate(wave, t1, t2, "squared")
        print(i1)
        plot_integral(wave, t1, t2, i1)
        i2 = integrate(wave, t2, t3, "squared")
        print(i2)
        plot_integral(wave, t2, t3, i2)

        i0ot = initial_pulse(frequency)

        i12 = i1 + i2
        ratio_unnormalized = i2 / i1
        i1ot = i1/(t2-t1)
        i2ot = i2/(t3-t2)
        i12ot = i12 / (t3-t1)
        ratio_normalized = i2ot / i1ot
        Q_scattering = (i1ot + i2ot) / i1ot
        Q_intrinsic = i0ot / (i0ot - i1ot - i2ot)


        # Save the plot and data
        plt.savefig('C:\\Users\\13764\\Documents\\ACADEMICS\\sr\\physics intern\\matec\\' + cube + '\\' + face + '\\' + frequency + ' MHz\\' + date + '\\' + spot)
        plt.clf()   # Clear to avoid overlap
        #data_row.extend([pulse_width, i1, i2, i1i2, ratio_unnormalized, Q_scattering, Q_intrinsic, i1ot, i2ot, i12ot, ratio_normalized])
        data_row.extend([pulse_width, i1, i2, ratio_unnormalized, Q_scattering, Q_intrinsic, i1ot, i2ot, ratio_normalized])

        data.append(data_row)

    # Write to "integrals.csv"
    csv_file_path = 'C:\\Users\\13764\\Documents\\ACADEMICS\\sr\\physics intern\\matec\\integrals_new_' + frequency + '_MHz.csv'
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)
    print(f'Integral written to {csv_file_path}')

#main("1")
#main("2.25")
main("5")
#main("10")