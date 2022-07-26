import matplotlib.pyplot as plt
import numpy as np


def dft(samples: np.ndarray, bins: int) -> np.ndarray:
    freq_components = np.zeros(bins, dtype=complex)
    for freq_bin in range(bins):
        freq_weight = 0
        freq = freq_bin / bins
        for sample_idx in range(samples.size):
            angle = 2 * np.pi * freq * sample_idx
            freq_weight += samples[sample_idx] * np.exp(-1j * angle)
        freq_components[freq_bin] = freq_weight
    return freq_components


if __name__ == "__main__":
    sample_rate = 10  # samples per unit time
    time = 10  # units of time
    t = np.arange(sample_rate * time)
    sine1 = 4 * np.sin(2 * np.pi * 0.10 * t)
    sine2 = np.sin(2 * np.pi * 0.24 * t)
    samples = sine1 + sine2
    plt.figure(0)
    plt.title("Samples")
    plt.plot(t, samples)
    frequency_components = dft(samples, sample_rate * time)
    plt.figure(1)
    plt.title("DFT")
    plt.plot(np.arange(0, sample_rate, sample_rate/100), np.abs(frequency_components))
    plt.figure(2)
    plt.title("NumPy FFT")
    frequency_components_np_fft = np.fft.fftshift(np.fft.fft(samples))
    plt.plot(np.arange(sample_rate/-2, sample_rate/2, sample_rate/100), np.abs(frequency_components_np_fft))
    plt.show()
