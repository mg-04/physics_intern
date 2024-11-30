# TO DO:    automate the algorithm
#           use energy (square)
#           record gain/attenuation constant, and number of cycles

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import csv

class data:
    def __init__(self, points, start, end):
        self.points = points
        self.start = start
        self.end = end
        self.length = end - start
    def find_peaks():
        pass
    

def extract_data(input):
    # organize the text file into separate graphs, based on line separation
    output = [] 
    temp = []
    for line in input:
        if line == "\n":
            output.append(temp)
            temp = []
            continue
        temp.append(line.strip())
    output.append(temp)
    
    # crops the header
    for i in [1]:
        graph = output[i]
        
        #print('Graph:', graph)
        length = int(graph[16]) # locates the length
        print(length)

        # cuts the data portion
        for j in range(17, 17+length):
            try:
                graph[j] = (int(graph[j]) - 128) / 128 * 100
            except:
                break
    output[0] = graph[17:(17+length)]

    return output

def find_peak_amp(data, start, end):
    # start, end time are in us. 
    start *= 100
    end *= 100
    t = start
    max_time = t
    max_amp = 0
    for point in data[start:end]:
        if np.abs(point) > max_amp:
            max_amp = np.abs(point)
            max_time = t
        t += 1
    return [max_time, max_amp]

def add(data1, data2, weight1=1, weight2=1):
    sum = []
    norm1 = weight1 / (weight1 + weight2)
    norm2 = weight2 / (weight1 + weight2)
    for idx, point in enumerate(data1):
        sum.append(norm1 * point)
    for idx, point in enumerate(data2):
        try:
            sum[idx] += norm2 * point
        except: 
            sum[idx] = norm2 * point
    return sum

def subtract(data1, data2):
    # data1: signal, data2: noise
    sum = []
    for idx, point in enumerate(data1):
        sum.append(point)
    for idx, point in enumerate(data2):
        try:
            sum[idx] -= point
        except: 
            break
    return sum

# reads a pre-written data file
def read(path):
    # output is a list of floats representing data
    file = open(path, 'r')
    length = int(float(file.readline().strip()))
    #print(length)
    data = []
    for i in range(length):
        data.append(float(file.readline().strip()))
    file.close()
    #print(data)
    return data

def write(path, data):
    for idx, number in enumerate(data):
        data[idx] = str(number) + '\n'
    file = open(path, 'w')
    file.writelines([str(len(data))+'\n'])
    file.writelines(data)
    file.close()

def increase_gain(waveform, old_gain_dB, new_gain_dB):
    # Convert dB to linear scale
    old_gain_linear = 10 ** (old_gain_dB / 10)
    new_gain_linear = 10 ** (new_gain_dB / 10)

    # Apply gain to each sample
    new_waveform = np.array(waveform) * np.sqrt(new_gain_linear / old_gain_linear)

    return new_waveform

def read_guess(cube, face, frequency, date, spot, plt_gain = 11, subtract_bg=True, silence=True):
    # Guess the file name (gain) and read whatever that does not produce an error
    file_extensions = ['', ' gain 0db', ' gain 5db', ' gain 11db', ' gain 16db', ' gain 21db', ' gain 26db', ' gain 0', ' gain 5', ' gain 11', ' gain 16', ' gain 21', ' gain 26', ' gain 41', ' g 41', ' g 51', 'g -10']
    for file_extension in file_extensions:
        try:
            file_name = 'C:\\Users\\13764\\Documents\\ACADEMICS\\physics intern\\matec\\' + cube + '\\' + face + '\\' + frequency + ' MHz\\' + date + '\\' + spot + file_extension + '.txt'                
            if not(silence):
                print("Trying", file_name)
            wave = read(file_name)
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError(f"Failed to locate file: {file_name}")
    
    # Guess gain
    try:    # if gain specified in the file name
        print(file_extension.split()[1])
        if ("db" in file_extension):
            wave_gain = int(file_extension.split()[1][:-2])
            
        else:
            wave_gain = int(file_extension.split()[1])
    except:     # default gain: -10 for 0.25 MHz, 11 for 1 MHz, 16 for 2.25, 51 for 5
        if frequency == '1':
            wave_gain = 11
        elif frequency == '2.25':
            wave_gain = 16
        else:
            wave_gain = 51

    # Guess bg gain
    if frequency == '0.25':
        background_gain = -10
    elif frequency == '1':
        background_gain = 11
    elif frequency == '2.25':
        if ((cube == '6-30-22 (fg)' and face != 'top')):
            background_gain = 26
        elif(cube == '8-23-22 (ds)' and face == 'top'):
            background_gain = 16
        else: background_gain = 21
    elif frequency == '5':
        if (("8-23-22 (ds)" == cube and date == "12-14-23") or cube == "2-7-23 (ds)"):
            background_gain = 41
        else:
            background_gain = 51
    elif frequency == '10':
        background_gain = 51

    # Guess bg name (as well)
    background = read('C:\\Users\\13764\\Documents\\ACADEMICS\\physics intern\\matec\\' + cube + '\\' + face + '\\' + frequency + ' MHz\\' + date + '\\background.txt') 

    # Modify wave and bg gain, and subtract
    background = increase_gain(background, background_gain, plt_gain)
    wave = increase_gain(wave, wave_gain, plt_gain)
    if subtract_bg:
        wave = subtract(wave, background)

    if not(silence):
        print("Wave db =", wave_gain, "Bg db =", background_gain)
    return wave

def integrate(data, start, end, mode, norm=True): 
    # data: list of floats
    # start, end: index of start/end
    # mode: string: "linear", "abs", "rms"
    # norm: boolean: whether to subtract the average while integrating

    # return: integrated value

    avg = 0        
    if(norm):   # normalize by subtracting the average of the whole wave
        avg = np.average(data)
    #print(avg)

    if mode == "linear":
        sum = 0
        for idx, point in enumerate(data[start: end], start):
            #print(data[idx])
            sum += 0.5*(data[idx]-avg+data[idx+1]-avg)
            #print(idx, idx+1, data[idx]-avg, data[idx+1]-avg,0.5*(data[idx]-avg+data[idx+1]-avg),sum)
        #print(total_sum)
        return sum
    
    elif mode == "abs":
    # trapezoidal integration of a wave
        sum = 0
        for idx, point in enumerate(data[start: end], start):
            #print(data[idx], sum)
            # when there is a sign change
            if (np.sign(data[idx]-avg) == -np.sign(data[idx+1]-avg) and np.sign(data[idx]-avg) != 0):
                # find where the wave crosses the x-axis
                cross = abs(data[idx]-avg) / (abs(data[idx+1]-avg)+abs(data[idx]-avg))
                # integrate left and right separately
                sum += 0.5 * (cross * np.abs(data[idx]-avg) + (1-cross) * np.abs(data[idx+1]-avg))
            # integrate normally
            else:
                sum += 0.5*(np.abs((data[idx]-avg)) + np.abs(data[idx+1]-avg))
        return sum
    
    elif mode == "rms": # same idea as the abs mode
        sum = 0
        for idx, point in enumerate(data[start: end], start):
            #print(data[idx], sum)
            # when there is a sign change
            if (np.sign(data[idx]-avg) == -np.sign(data[idx+1]-avg) and np.sign(data[idx]-avg) != 0):
                # find where the wave crosses the x-axis
                cross = abs(data[idx]-avg) / (abs(data[idx+1]-avg)+abs(data[idx]-avg))
                # integrate left and right separately, quadratic is 1/3
                sum += 1/3 * (cross * ((data[idx]-avg)**2) + (1-cross) * ((data[idx+1]-avg)**2))
            # integrate normally
            else:
                sum += 0.5*(((data[idx]-avg)**2)+((data[idx+1]-avg)**2))
        return np.sqrt(sum)
    else:
        raise TypeError("integrate: Invalid Mode")


def old_integrate(data, start, end, norm=True):
    # data: list of floats
    # start/end: starting and ending time in us, inclusive
    # norm: whether to subtract the average of the whole integral
    # trapezoidal integration of a wave, index are in us.
    sum = 0
    for idx, point in enumerate(data[start: end-1], start):
        #print(data[idx])
        sum += 0.5*(data[idx]+data[idx+1])
    total_sum = 0

    if(norm):
        for idx, point in enumerate(data[:len(data)-1], 0):
            total_sum += 0.5*(data[idx]+data[idx+1])
    return sum - total_sum

def old_integrate_abs(data, start, end, norm=True):    # include start and end. \int(start, end). 
    # trapezoidal integration of a wave
    avg = 0

    #print(data)
    #print(start, end)
    
    if(norm):
        avg = old_integrate(data, 0, len(data), False) / len(data)

    sum = 0
    for idx, point in enumerate(data[start: end-1], start):
        #print(data[idx], sum)
        # when there is a sign change
        if (np.sign(data[idx]-avg) == -np.sign(data[idx+1]-avg) and np.sign(data[idx]-avg) != 0):
            # find where the wave crosses the x-axis
            cross = abs(data[idx]-avg) / (abs(data[idx+1]-avg)+abs(data[idx]-avg))
            # integrate left and right separately
            sum += 0.5 * (cross * np.abs(data[idx]-avg) + (1-cross) * np.abs(data[idx+1]-avg))
        # integrate normally
        else:
            sum += np.abs(0.5*(data[idx]-avg+data[idx+1]-avg))
    
    return sum

def find_local_maxima(wave):
    local_maxima_indices = []

    for i in range(10, len(wave) - 10):
        if all(wave[i] > wave[j] for j in range(i - 10, i)) and all(wave[i] > wave[j] for j in range(i + 1, i + 11)):
            local_maxima_indices.append(i)

    return local_maxima_indices

def nfind_peak(data, amp_threshold=10):
    # returns a tuple of all peaks above a certain ampliude threshold
    peaks = []  # a list containing peaks. Each peak in the form of (t, amp) tuple
    

    for i in range(10, len(data) - 10):
        if all(data[i] >= data[j] for j in range(i - 10, i)) and all(data[i] >= data[j] for j in range(i + 1, i + 11)) and data[i] > amp_threshold:
            peaks.append((i, data[i]))
        if all(data[i] <= data[j] for j in range(i - 10, i)) and all(data[i] <= data[j] for j in range(i + 1, i + 11)) and data[i] < -amp_threshold:
            peaks.append((i, data[i]))
    #print(peaks)
    
    filtered_peaks = []
    start = -1 
    prev_peak = peaks[0]
    for idx, peak in enumerate(peaks[1:], 1):  
        #print(idx, peak, start)
        if peak[1] == prev_peak[1] and peak[0] == prev_peak[0]+1:
            #print("equal")
            if (start==-1):
                start = idx-1
        else:
            if(start != -1):
                filtered_peaks.append(peaks[(start+idx-1)//2])
                filtered_peaks.append(peak)
                start = -1
            else:
                filtered_peaks.append(peak)

        prev_peak = peak

    #print(filtered_peaks)
    return filtered_peaks
def find_peak(data, amp_threshold=10):
    # returns a tuple of all peaks above a certain ampliude threshold
    peaks = []  # a list containing peaks. Each peak in the form of (t, amp) tuple
    prev_2 = data[0]
    prev_1 = data[1]
    max = False
    min = False
    for idx, cur in enumerate(data[2:], 2):
        #print(idx, prev_2, prev_1, cur)
        if (cur < prev_1):  # max
            #print(cur)
            if(prev_1>0):
                if (prev_1 > prev_2):   # distinct peak
                    peaks.append((idx-1, prev_1))
                if (prev_1 == prev_2 and max):   # end of a plateau
                    #print(start, idx - 1)
                    peaks.append(((start + idx-1) // 2, data[(start + idx-1) // 2]))  # append the average

        if (cur > prev_1):  # min
            #print(cur, max)
            if(prev_1<0):
                if (prev_1 < prev_2):   # distinct peak
                    peaks.append((idx-1, prev_1))
                if (prev_1 == prev_2 and min):   # end of a plateau
                    peaks.append(((start + idx-1) // 2, data[(start + idx-1) // 2]))  # append the average

        # plateau tracker
        if (cur == prev_1):
            if (prev_2 < prev_1):  # if it's the start of plateau
                start = idx - 1     # mark the starting point as prev_1
                max = True          # avoid tracking "points of inflection"
                min = False
            if (prev_2 > prev_1):
                start = idx - 1
                min = True
                max = False   
        prev_2 = prev_1
        prev_1 = cur
    
    peaks = [peak for peak in peaks if np.abs(peak[1]) > amp_threshold]
    filtered_peaks = []
    start = -1 
    prev_peak = peaks[0]

    # filter out close peaks that have the same value but are 1 units apart, using the similar find_zero method
    for idx, peak in enumerate(peaks[1:], 1):  
        #print(idx, peak, start)
        if peak[1] == prev_peak[1] and peak[0] == prev_peak[0]+1:
            print("equal")
            if (start==-1):
                start = idx-1
        else:
            if(start != -1):
                filtered_peaks.append(peaks[(start+idx-1)//2])
                filtered_peaks.append(peak)
                start = -1
            else:
                filtered_peaks.append(peak)

        prev_peak = peak

    #print(filtered_peaks)
    return filtered_peaks

def summary(peaks, head = 50):
    # prints a summary of the peak gaps, from high to low
    # peaks: a list of tuples (t, A)
    # head: how many elements to print
    gaps = []
    prev_peak = peaks[0]
    for peak in peaks:
        gap = peak[0] - prev_peak[0]
        gaps.append((gap, prev_peak, peak))
        prev_peak = peak
    sorted_gaps = sorted(gaps, key=lambda x:-x[0])
    sorted2 = []
    for gap in gaps:
        if gap[0] > head:
            sorted2.append(gap)
    #print(sorted(sorted2, key=lambda x:x[1][0]))
    return sorted(sorted2, key=lambda x:x[1][0])

    # print the ones more than 50? Since it's 1 MHz

def find_zeroes(data):
    # returns a tuple of all peaks above a certain ampliude threshold
    zeroes = []
    start = -1  # starting index of a zero sequence, if any
    prev_value = data[0]
    for idx, value in enumerate(data[1:], 1):

        # gradual zero crossing
        if (value == 0):
            if(start == -1):    # beginning of a new zero sequence
                start = idx
        if (value != 0):    # end of a sequence of zeroes
            if (start != -1):   # have a recorded sequence of zeros already
                zeroes.append(((start+idx-1)//2, data[(start+idx-1)//2])) # append the average
                start = -1  # reset the start
        
        # sudden zero crossing: no zero values, but a sign change
        if ((prev_value > 0 and value < 0) or (prev_value < 0 and value > 0)):
            zeroes.append((idx, data[idx]))
        prev_value = value
    #print(zeroes)
    return zeroes

def search(t, direction, mode, peaks, zeroes):
    # point: t
    # dirction: string "peak" or "zero"
    # mode: string "left" or "right"
    # peaks, zeroes: list of peaks and zeroes
    if (mode == "peak"):
        arr = peaks
    elif (mode == "zero"):
        arr = zeroes
    else: 
        raise TypeError("search: Invalid Mode")

    low, high = 0, len(arr) - 1
    result = None
    
    if(direction == "left"):  # 
        while low <= high:
            mid = (low + high) // 2
            if arr[mid][0] < t:
                result = arr[mid]
                low = mid + 1
            else:
                high = mid - 1
        if result == None:
            raise ValueError("search: Not Found")
        return result[0]
    elif (direction == "right"):
        while low <= high:
            #print(low, high)
            mid = (low + high) // 2
            #print(arr[mid][0])
            if arr[mid][0] > t:
                result = arr[mid]
                high = mid - 1
            else:
                low = mid + 1
        if result == None:
            raise ValueError("search: Not Found")
        return result[0]
    else:
        raise TypeError("search: Invalid Direction")

def initial_pulse(frequency):
    # returns data (i0ot) about the initial pulse of a specific frequency:
    if frequency == '1':
        i0 = 9777.09094869677
        t0 = 46
        t05 = 653
    elif frequency == '2.25':
        i0 = 5429.48045129347
        t0 = 45
        t05 = 424
    elif frequency == '5':
        i0 = 4136.11522740971
        t0 = 46
        t05 = 228
    elif frequency == '10':
        i0 = 1206.88673746273
        t0 = 45
        t05 = 99
    else:
        raise KeyError("Unknown frequency. Input type must be STR")
    i0ot = np.sqrt(i0**2 / (t05-t0))
    return i0ot

def tplot(data):
    x =np.arange(0, 0.01* len(data), 0.01)
    plt.plot(x, np.array(data).astype(float))
    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude (%)")
    plt.ylim([-100, 100])
    plt.show()

def plot_point(wave, target, annotation='', y_pos = 20):    
    plt.scatter(target, wave[target], c='red')
    plt.annotate(f'{annotation} ({target}, {wave[target]:.2f})',
                 xy=(target, wave[target]),
                 xytext=(target, wave[target] + y_pos),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center')
    
def plot_integral(wave, start, end, value, y_pos = 20, annotation=''):
    plt.axvline(start, color='red', linestyle='--', label='Start of Interval')
    plt.axvline(end, color='red', linestyle='--', label='End of Interval')
    center_interval = (start + end) / 2
    plt.text(center_interval, y_pos, f'{annotation}{value:.1f}', color='red', ha='center', va='bottom')

def fourier(input, dt=10e-9):
    fft = np.fft.rfft(input)
    freq = np.fft.rfftfreq(len(input), dt)
    plt.plot(freq/1e6,np.abs(fft)/np.max(np.abs(fft)),linewidth=.75, label='Dry Coupling Shear Waves')
    plt.title("FFT Comparison")
    plt.ylabel("Normalised Amplitude")
    plt.xlabel("Frequency [MHz]")
    plt.legend(loc="upper right")
    plt.xlim(0,5)
    plt.ylim(0,1)
    plt.show()
    return (fft, freq)