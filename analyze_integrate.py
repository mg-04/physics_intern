# auto_analyze_integrate.py
from wave_util import *
import sys
import csv
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_background(folder: Path, background_gain_db: float, target_gain_db: float) -> np.ndarray:
    """Locate and load background trace, normalized to target gain."""
    suffixes = ['', f' gain {int(background_gain_db)}db', f' {int(background_gain_db)}db']
    for suf in suffixes:
        path = folder / f"background{suf}.txt"
        if path.exists():
            bg = read_data_file(path)
            return adjust_gain(bg, background_gain_db, target_gain_db)
    raise FileNotFoundError(f"Background file not found under {folder}")

def guess_wave(folder: Path, name: str, target_gain_db: float) -> Tuple[np.ndarray, float]:
    """Attempt to load a waveform by trying gain suffixes; return data and original gain."""
    suffixes = [
        '', ' gain 0db',' gain 5db',' gain 11db',' gain 16db',' gain 21db',' gain 26db',
        ' gain 0',' gain 5',' gain 11',' gain 16',' gain 21',' gain 26',
        ' gain 41',' g 41',' g 51',' g -10'
    ]
    for suf in suffixes:
        path = folder / f"{name}{suf}.txt"
        if not path.exists():
            continue
        wave = read_data_file(path)
        # parse numeric gain
        parts = suf.strip().split()
        if len(parts) >= 2:
            gain_str = parts[-1].replace('db','')
            try:
                orig_gain = float(gain_str)
            except ValueError:
                orig_gain = target_gain_db
        else:
            orig_gain = target_gain_db
        # normalize
        wave = adjust_gain(wave, orig_gain, target_gain_db)
        return wave, orig_gain
    raise FileNotFoundError(f"Wave file '{name}' not found under {folder}")

def plot_integral_area(wave: np.ndarray, start: int, end: int, y0: float = 0.0, **fill_kwargs):
    """
    Fill area under wave between start and end, anchored at y0.
    """
    x = np.arange(start, end+1)
    y = wave[start:end+1]
    plt.fill_between(x, y, y0, **fill_kwargs)

def run_manual(args):
    """Interactive, file-by-file analysis."""
    folder = Path(args.folder)
    bg = load_background(folder, args.background_gain, args.plt_gain)
    t1 = args.t1
    results = []
    while True:
        name = input("File name (or 'quit'): ")
        if name.lower() == 'quit':
            break
        try:
            wave, orig_gain = guess_wave(folder, name, args.plt_gain)
        except FileNotFoundError as e:
            print(e)
            continue
        data_row = [name]
        print(f"Wave gain={orig_gain}dB, BG gain={args.background_gain}dB")
        # subtract background
        wave = subtract_signals(wave, bg)
        # find events for plotting
        peaks = find_peaks(wave, amp_threshold=10.0) 
        zeroes = find_zero_crossings(wave)
        maximum = wave.max()
        # initial plot and annotations
        plt.figure()
        plt.plot(wave[:4000], label='Waveform')
        for i in range(200):
            idx = binary_search_event(peaks if i%2==0 else zeroes,
                                      target_idx=i*args.event_step,
                                      direction='right')
            plot_event(wave, idx, label=None,
                       y_offset=((-1)**i)*maximum/15)
        plt.legend()
        plt.show()
        print("Enter integration windows:")
        while True:
            t2 = int(input("t2: "))
            t3 = int(input("t3: "))
            i1 = integrate_signal(wave, t1, t2, 'rms')
            i2 = integrate_signal(wave, t2, t3, 'rms')
            # plot integrals
            plt.figure()
            plt.plot(wave[:4000], label='Waveform')
            plot_integral_area(wave, t1, t2, i1, y_pos=maximum/2)
            plot_integral_area(wave, t2, t3, i2, y_pos=maximum/2)
            plt.legend()
            plt.show()
            if input("Confirm? [Y]/N ") in ['', 'Y','y']:
                break
        # compute metrics
        ratio = i2/i1
        i1ot = np.sqrt(i1**2/(t2-t1))
        i2ot = np.sqrt(i2**2/(t3-t2))
        ratio_norm = i2ot/i1ot
        data_row += [t1, t2, t3, i1, i2, ratio, i1ot, i2ot, ratio_norm]
        results.append(data_row)
    # write CSV
    out = folder / 'integrals.csv'
    with out.open('w', newline='') as f:
        csv.writer(f).writerows(results)
    print(f"Results saved to {out}")

def run_batch(args):
    freq = args.frequency
    base = Path(args.base_dir)
    # read times CSV
    rows = list(csv.reader(open(args.times_csv)))
    header, data_rows = rows[0], rows[1:]
    results = []
    for row in data_rows:
        if not row[3]:
            results.append([])
            continue
        cube = row[0] or cube
        face = row[1] or face
        date = row[2] or date
        spot = row[3]
        t1, t2, t3 = map(int, row[4:7])
        folder = base / cube / face / f"{freq} MHz" / date
        wave = guess_wave(folder, spot, args.target_gain)
        # subtract background implicitly
        # plot and compute integrals
        xlim = args.xlim
        i1 = integrate_signal(wave, t1, t2, mode='squared')
        i2 = integrate_signal(wave, t2, t3, mode='squared')
        # metrics
        i0ot = initial_pulse_info(freq)
        ratio = i2 / i1
        i1ot = i1 / (t2 - t1)
        i2ot = i2 / (t3 - t2)
        i12ot = (i1 + i2) / (t3 - t1)
        ratio_norm = i2ot / i1ot
        Q_scat = (i1ot + i2ot) / i1ot
        Q_int = i0ot / (i0ot - i1ot - i2ot)
        # save plot
        plt.figure()
        plt.plot(wave[:xlim], label='Waveform')
        plt.savefig(folder / f"{spot}.png"); plt.clf()
        results.append([cube, face, date, spot, t1, t2, t3,
                        i1, i2, ratio, Q_scat, Q_int, i1ot, i2ot, i12ot, ratio_norm])
    out_csv = base / f"integrals_{freq}MHz.csv"
    with open(out_csv, 'w', newline='') as f:
        csv.writer(f).writerows(results)
    print(f"Batch results saved to {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Wave analysis tool")
    sub = parser.add_subparsers(dest='mode', required=True)

    m = sub.add_parser('manual', help='Interactive analysis')
    m.add_argument('-f','--folder', required=True, help='Data folder')
    m.add_argument('-b','--background-gain', type=float, required=True)
    m.add_argument('-g','--plt-gain', type=float, default=11.0)
    m.add_argument('-t','--t1', type=int, required=True, help='Start index for integration')

    b = sub.add_parser('batch', help='Batch CSV-driven analysis')
    b.add_argument('-q','--frequency', required=True, help='Frequency MHz')
    b.add_argument('-i','--times-csv', required=True, help='CSV of integral times')
    b.add_argument('-g','--plt-gain', type=float, default=11.0)
    b.add_argument('--base-dir', default="C:/Users/13764/Documents/ACADEMICS/sr/physics intern/matec")
    b.add_argument('--xlim', type=int, default=None, help='X-axis limit for plots')

    args = parser.parse_args()
    if args.mode == 'manual':
        run_manual(args)
    else:
        run_batch(args)

if __name__ == '__main__':
    main()