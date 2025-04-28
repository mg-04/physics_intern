# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 23:24:35 2023

@author: neelesh bhadwal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Import Data'''

'''Data Set 1'''
raw_data=pd.read_csv(r"C:\Users\neele\Desktop\processed.csv",header=0,dtype = {"10805":"float64"})  #1\retest cooling ts #1 after receive from soundbite\Cooling TS stack #1 second reception from soundbite.csv",header=20,dtype = {"TIME":"float64",'CH1':"float64"})
"""Time interval is dt"""
dt=10e-9
total_rows=raw_data.shape[0]
voltage=raw_data.iloc[1:total_rows,0]
rate=1/dt
time=np.zeros(total_rows-1)
for count in np.arange(1,total_rows-1):
    time[count]=10e-9*count

"""Plot Signal"""
plt.figure(1)
plt.plot(time,voltage,linewidth=.75,label='Dry Coupling Shear Waves')
plt.title('Shear Waves')
plt.xlabel('Time (s)')
plt.ylabel('Volts (V)')
plt.xlim(0,10e-5)
plt.legend(loc="upper right")
plt.show()

"""FFT"""

'''Data Set 1'''
fft=np.fft.rfft(voltage)
freq=np.fft.rfftfreq(total_rows,dt)

"""Plot FFT"""
plt.figure(2)
plt.plot(freq/1e6,np.abs(fft)/np.max(np.abs(fft)),linewidth=.75, label='Dry Coupling Shear Waves')
plt.title("FFT Comparison")
plt.ylabel("Normalised Amplitude")
plt.xlabel("Frequency [MHz]")
plt.legend(loc="upper right")
plt.xlim(0,5)
plt.ylim(0,1)
plt.show()

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("Signal Analysis")
ax1.plot(time,voltage,linewidth=.75)
ax1.set_ylabel('Signal (Volts)')
ax1.set_xlabel('Time (s)')
ax1.set_xlim([0, 10e-5])
ax1.grid()

ax2.plot(freq/1e6,np.abs(fft)/np.max(np.abs(fft)),linewidth=.75)
ax2.set_xlim([0, 3e6])
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Normalised Amplitude')
ax2.set_xlim([0,2])
ax2.set_ylim([0,1])
ax2.grid()
"""You can cahnge the signal window you are looking at"""
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("Signal Analysis")
ax1.plot(time[3000:4500],voltage[3000:4500],linewidth=.75)
ax1.set_ylabel('Signal (Volts)')
ax1.set_xlabel('Time (s)')
ax1.set_xlim([3e-5, 4.5e-5])
ax1.grid()

fft=np.fft.rfft(voltage[3000:4500])
freq=np.fft.rfftfreq(len(voltage[3000:4500]),dt)

ax2.plot(freq/1e6,np.abs(fft)/np.max(np.abs(fft)),linewidth=.75)
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Normalised Amplitude')
ax2.set_xlim([0,2])
ax2.set_ylim([0,1])
ax2.grid()



