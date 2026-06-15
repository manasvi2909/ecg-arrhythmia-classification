import os
import numpy as np
import wfdb

# Folders
RAW_DIR = os.path.join("data", "raw", "mit-bih")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Records to use (start with a few)
RECORDS = [
    "100", "101", "102", "103", "104",
    "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115",
    "118", "119",
    "200", "201", "202", "203",
    "207", "210", "212", "213"
]

# Map original MIT-BIH beat symbols to 5 classes: N, S, V, F, Q
AAMI_MAP = {
    # N class
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    # S class
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    # V class
    "V": "V",
    "E": "V",

    # F class
    "F": "F",

    # Q class
    "/": "Q",
    "f": "Q",
    "Q": "Q",
    "P": "Q",
}


def map_symbol(symbol):
    return AAMI_MAP.get(symbol, None)


def process_record(record_name):
    record_path = os.path.join(RAW_DIR, record_name)

    # Read signal
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]  # first channel

    # Read annotations
    ann = wfdb.rdann(record_path, "atr")
    r_peaks = ann.sample
    symbols = ann.symbol

    beats = []
    labels = []

    before = 180
    after = 180
    total_len = before + after

    for peak, symbol in zip(r_peaks, symbols):
        label = map_symbol(symbol)
        if label is None:
            continue

        start = peak - before
        end = peak + after

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]
        if len(beat) != total_len:
            continue

        beats.append(beat)
        labels.append(label)

    return beats, labels


def main():
    all_beats = []
    all_labels = []

    for rec in RECORDS:
        print(f"Processing record {rec}...")
        beats, labels = process_record(rec)
        print(f"  Found {len(beats)} beats")
        all_beats.extend(beats)
        all_labels.extend(labels)

    X = np.array(all_beats, dtype=np.float32)
    y = np.array(all_labels)

    print("Final X shape:", X.shape)
    print("Final y shape:", y.shape)

    np.save(os.path.join(PROCESSED_DIR, "X.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "y.npy"), y)
    print("Saved X.npy and y.npy in data/processed/")


if __name__ == "__main__":
    main()