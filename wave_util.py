import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Union

def extract_segments(lines: List[str]) -> List[List[str]]:
    """
    Split raw text lines into segments separated by blank lines.
    """
    segments, current = [], []
    for line in lines:
        if line.strip() == "":
            if current:
                segments.append(current)
                current = []
        else:
            current.append(line.strip())
    if current:
        segments.append(current)
    return segments

def parse_graph_segment(segment: List[str], header_offset: int = 16) -> np.ndarray:
    """
    Parse a single graph segment: interpret header for data length,
    convert samples to percent deviation.
    """
    try:
        length = int(segment[header_offset])
    except (IndexError, ValueError):
        raise ValueError("Invalid segment header for length")
    data_lines = segment[header_offset + 1 : header_offset + 1 + length]
    values = []
    for line in data_lines:
        try:
            raw = int(line)
            pct = (raw - 128) / 128 * 100
            values.append(pct)
        except ValueError:
            break
    return np.array(values, dtype=float)

def find_peak_amplitude(signal: np.ndarray, start_us: float, end_us: float, sample_rate: float = 100e6) -> Tuple[float, float]:
    """
    Find the time (us) and amplitude (%) of the maximum absolute peak
    between start_us and end_us.
    """
    start_idx = int(start_us * (sample_rate / 1e6))
    end_idx = int(end_us * (sample_rate / 1e6))
    window = signal[start_idx:end_idx]
    if window.size == 0:
        raise ValueError("Empty signal window for peak search")
    peak_idx = np.argmax(np.abs(window))
    amp = np.abs(window[peak_idx])
    time_us = (start_idx + peak_idx) / (sample_rate / 1e6)
    return time_us, amp

def weighted_sum(sig1: np.ndarray, sig2: np.ndarray, w1: float = 1.0, w2: float = 1.0) -> np.ndarray:
    """
    Return w1/(w1+w2)*sig1 + w2/(w1+w2)*sig2, handling different lengths.
    """
    total = w1 + w2
    norm1, norm2 = w1 / total, w2 / total
    max_len = max(sig1.size, sig2.size)
    out = np.zeros(max_len, dtype=float)
    out[: sig1.size] += norm1 * sig1
    out[: sig2.size] += norm2 * sig2
    return out

def subtract_signals(sig: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """
    Subtract noise from signal (zero-extended to match lengths).
    """
    length = max(sig.size, noise.size)
    padded_sig = np.pad(sig, (0, length - sig.size), constant_values=0)
    padded_noise = np.pad(noise, (0, length - noise.size), constant_values=0)
    return padded_sig - padded_noise

def read_data_file(path: Union[str, Path]) -> np.ndarray:
    """
    Read a text file: first line is length, then that many floats.
    """
    p = Path(path)
    with p.open() as f:
        length = int(float(f.readline().strip()))
        data = [float(f.readline().strip()) for _ in range(length)]
    return np.array(data, dtype=float)

def write_data_file(path: Union[str, Path], data: Union[List[float], np.ndarray]) -> None:
    """
    Write data to file: length on first line, then one value per line.
    """
    p = Path(path)
    arr = np.asarray(data, dtype=float)
    with p.open('w') as f:
        f.write(f"{arr.size}\n")
        for val in arr:
            f.write(f"{val}\n")

def adjust_gain(
    waveform: np.ndarray, old_gain_db: float, new_gain_db: float
) -> np.ndarray:
    """
    Adjust waveform amplitude from old_gain_db to new_gain_db (dB).
    """
    old_lin = 10 ** (old_gain_db / 10)
    new_lin = 10 ** (new_gain_db / 10)
    factor = np.sqrt(new_lin / old_lin)
    return waveform * factor

def read_guess(cube: str, face: str, frequency: float, date: str, spot: str,
    target_gain_db: float = 11.0, subtract_bg: bool = True, silent: bool = True, 
    base_dir: Union[str, Path] = "C:/Users/13764/Documents/ACADEMICS/sr/physics intern/matec") -> np.ndarray: 
    """
    Locate and load a waveform file by guessing gain extensions,
    normalize its gain to `target_gain_db`, optionally subtract background.

    Parameters:
      cube, face, frequency, date, spot: path components to file
      target_gain_db: desired gain to normalize both wave and background
      subtract_bg: whether to subtract background trace
      silent: if False, prints diagnostic messages
      base_dir: root directory for file hierarchy

    Returns:
      Processed waveform as a 1D numpy array
    """
    base_path = Path(base_dir) / cube / face / f"{frequency} MHz" / date

    gain_suffixes = [
        "",
        " gain 0db", " gain 5db", " gain 11db", " gain 16db", " gain 21db", " gain 26db",
        " gain 0",   " gain 5",   " gain 11",   " gain 16",   " gain 21",   " gain 26",
        " gain 41",  " g 41",     " g 51",     " g -10"
    ]

    wave = None
    wave_gain = None
    # try reading waveform file
    for suffix in gain_suffixes:
        fname = f"{spot}{suffix}.txt"
        path = base_path / fname
        if not path.exists():
            continue
        if not silent:
            print(f"Loading waveform: {path}")

        try: 
            wave = read_data_file(path)
            # extract numeric gain from file suffix
            tokens = suffix.strip().split()
            if len(tokens) >= 2:
                gain_str = tokens[-1].replace('db', '')
                wave_gain = float(gain_str)
            break
        except Exception as e:
            if not silent:
                print(f"Failed to read {path}: {e}")
            continue          
    if wave is None:
        raise FileNotFoundError(f"No waveform file found under {base_path} for spot '{spot}'")
    
    # set default gain if not parsed
    if wave_gain is None:
        default_gain_map = {0.25: -10.0, 1.0: 11.0, 2.25: 16.0, 5.0: 51.0, 10.0: 51.0}
        wave_gain = default_gain_map.get(frequency, target_gain_db)
        if not silent:
            print(f"Using default wave_gain={wave_gain} dB")
    
    # determine background gain based on frequency and context
    bg_gain_map = {
        0.25: -10.0,
        1.0: 11.0,
        2.25: 21.0,
        5.0: 51.0,
        10.0: 51.0
    }
    background_gain = bg_gain_map.get(frequency, target_gain_db)

    # read background trace
    bg_path = base_path / "background.txt"
    if not bg_path.exists():
        raise FileNotFoundError(f"Background file not found: {bg_path}")
    background = read_data_file(bg_path)

    # normalize gains
    wave = adjust_gain(wave, wave_gain, target_gain_db)
    background = adjust_gain(background, background_gain, target_gain_db)

    if subtract_bg:
        wave = subtract_signals(wave, background)

    if not silent:
        print(f"Normalized wave to {target_gain_db} dB (orig {wave_gain} dB), ")
        print(f"bg gain {background_gain} dB")

    return wave

def integrate_signal(
    data: np.ndarray,
    start: int,
    end: int,
    mode: str = 'linear',
    normalize: bool = True
) -> float:
    """
    Integrate signal between indices [start, end] by mode:
      - 'linear': trapezoid
      - 'abs': trapezoid of abs deviations
      - 'squared': trapezoid of squared deviations
    If normalize, subtract average of tail (from index 1000 onward).
    """
    if normalize:
        avg = np.mean(data[1000:])
    else:
        avg = 0.0
    segment = data[start : end + 1] - avg
    if mode == 'linear':
        vals = (segment[:-1] + segment[1:]) / 2
    elif mode == 'abs':
        vals = (np.abs(segment[:-1]) + np.abs(segment[1:])) / 2
    elif mode == 'squared':
        vals = (segment[:-1]**2 + segment[1:]**2) / 2
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.sum(vals)

def find_local_maxima(signal: np.ndarray, window: int = 10) -> List[int]:
    """
    Return indices where signal[i] is greater than its `window` neighbors.
    """
    maxima = []
    n = signal.size
    for i in range(window, n - window):
        if np.all(signal[i] > signal[i-window:i]) and np.all(signal[i] > signal[i+1:i+window+1]):
            maxima.append(i)
    return maxima

def find_peaks(
    data: np.ndarray, amp_threshold: float = 10.0
) -> List[Tuple[int, float]]:
    """
    Find all local maxima and minima exceeding amp_threshold.
    """
    peaks = []
    # Use sign changes on derivative
    deriv = np.diff(data)
    zeros = np.where(deriv[:-1] * deriv[1:] < 0)[0] + 1
    for idx in zeros:
        amp = data[idx]
        if abs(amp) > amp_threshold:
            peaks.append((idx, amp))
    return peaks

def summarize_peaks(
    peaks: List[Tuple[int, float]], min_gap: int = 50
) -> List[Tuple[int, Tuple[int, float], Tuple[int, float]]]:
    """
    Return (gap, prev_peak, current_peak) for gaps >= min_gap, sorted by time.
    """
    results = []
    for (t0, a0), (t1, a1) in zip(peaks, peaks[1:]):
        gap = t1 - t0
        if gap >= min_gap:
            results.append((gap, (t0, a0), (t1, a1)))
    return sorted(results, key=lambda x: x[1][0])

def find_zero_crossings(data: np.ndarray) -> List[Tuple[int, float]]:
    """
    Detect zero-value samples or sign changes; return midpoint index and value.
    """
    crossings = []
    n = data.size
    # exact zeros
    zero_idxs = np.where(data == 0)[0]
    if zero_idxs.size > 0:
        # cluster contiguous zeros
        groups = np.split(zero_idxs, np.where(np.diff(zero_idxs) != 1)[0] + 1)
        for grp in groups:
            mid = grp[len(grp)//2]
            crossings.append((mid, data[mid]))
    # sign changes
    sign_changes = np.where(data[:-1] * data[1:] < 0)[0]
    for idx in sign_changes:
        crossings.append((idx+1, data[idx+1]))
    return sorted(crossings, key=lambda x: x[0])

def binary_search_event(
    events: List[Tuple[int, float]],
    target_idx: int,
    direction: str = 'left'
) -> int:
    """
    Find nearest event index < or > target_idx via binary search.
    """
    low, high = 0, len(events) - 1
    result = None
    key = lambda ev: ev[0]
    if direction == 'left':
        while low <= high:
            mid = (low + high) // 2
            if key(events[mid]) < target_idx:
                result = events[mid]
                low = mid + 1
            else:
                high = mid - 1
    elif direction == 'right':
        while low <= high:
            mid = (low + high) // 2
            if key(events[mid]) > target_idx:
                result = events[mid]
                high = mid - 1
            else:
                low = mid + 1
    else:
        raise ValueError("direction must be 'left' or 'right'")
    if result is None:
        raise ValueError("No matching event found")
    return result[0]

def initial_pulse_info(frequency: str) -> float:
    """
    Compute I0ot constant based on known pulse parameters per frequency.
    """
    params = {
        '1':    (9777.09094869677, 46, 653),
        '2.25': (5429.48045129347, 45, 424),
        '5':    (4136.11522740971, 46, 228),
        '10':   (1206.88673746273, 45, 99),
    }
    try:
        i0, t0, t05 = params[frequency]
    except KeyError:
        raise ValueError("Unknown frequency")
    return np.sqrt(i0**2 / (t05 - t0))

def tplot(data):
    x =np.arange(0, 0.01* len(data), 0.01)
    plt.plot(x, np.array(data).astype(float))
    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude (%)")
    plt.ylim([-100, 100])
    plt.show()

def plot_time_trace(data: np.ndarray, dt_us: float = 0.01, ylim: Tuple[float, float] = (-100, 100)) -> None:
    """
    Plot amplitude (%) vs time (us).
    """
    t = np.arange(data.size) * dt_us
    plt.plot(t, data)
    plt.xlabel("Time (μs)")
    plt.ylabel("Amplitude (%)")
    plt.ylim(ylim)
    plt.show()

def plot_event(wave: np.ndarray, idx: int, label: Optional[str] = None, y_offset: float = 20.0) -> None:
    """
    Scatter and annotate a single event on the waveform.
    """
    plt.scatter(idx, wave[idx], color='red')
    text = f"{label} ({idx}, {wave[idx]:.2f})" if label else f"({idx}, {wave[idx]:.2f})"
    plt.annotate(text,
                 xy=(idx, wave[idx]),
                 xytext=(idx, wave[idx] + y_offset),
                 arrowprops=dict(arrowstyle='->'))

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