# Wave Analysis Codes 
## Versions
1. Original (stable)
2. Refactored (April 2025)

# Workflow
## 1. Preprocessing
Matec outputs a `.txt` dump where
- Lines 1-16 are metadata
- Subsequent lines are sample values (0-255)

`parse_graph_segment()` parses the raw `.txt` dumps and transforms every value to [-100, 100] and writes back via `write_data_file()`
- From now on, all I/O uses the clean, normalized files through  `read_data_file()`
- The file paths embed frequency, spot, and gain
- `read_guess()` applies default gains per frequency but will override if a suffix explicitly specifies the gain.

## 2. Waveform Analysis
Each waveform can be analyzed by the following functions:
- `find_local_maxima()`
- `find_peaks()` above an amplitude threshold
- `find_zero_crossings()`
- `integrate()` over a time window
- `fourier()`

## 3. Locate and Integrate Reflections
The waveform and the peaks/zeroes are plotted. `analyze_integrate.py` lists their coordinates. Based on these, the user selects the appropriate window for the initial, scattering, and pulse regions
- `t1`: start of the scattering region. This is a predefined delay per frequency to skip the initial noise spike. 
- `t2`: start of the pulse region (end of scattering): manually input by the user, based on the first peak after the expected arrival time, above 10% for 1 MHz. If the pulse amplitude is below that, an interval is manually assigned
- `t3`: end of the pulse region: manually input by the user, based on the pulse annotated dots after the main pulse peak

Each region is then integrated. The ratios and $Q_Z$ are also computed.

## 4. Saving the result
All results are saved to `integrals.csv`, which can be further read and plotted