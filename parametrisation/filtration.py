import numpy as np
from scipy.signal import butter, filtfilt
from data_presentation.player import AudioPlayer
import matplotlib.pyplot as plt


class Filter:

    def lowpass_filter(self, data, cutoff, fs, order):
        normal_cutoff = cutoff / (fs/2)
        b, a = butter(order, normal_cutoff, btype='low', output='ba')
        y = filtfilt(b, a, data)
        return np.asarray(y, dtype=np.int16)

    def bandpass_filter(self, data, cutoffs, fs, order):
        low, high = cutoffs[0] / (fs / 2), cutoffs[1] / (fs / 2)
        b, a = butter(order, [low, high], btype='band', output='ba')
        y = filtfilt(b, a, data)
        return np.asarray(y, dtype=np.int16)

    def no_filter(self, data, cutoffs=None, fs=None, order=None):
        """function needed for ml_manager architecture"""
        return data

    def white_noise(self, data, noise_amp):
        noise_amp = max(data) * noise_amp
        noise = np.random.normal(0, noise_amp, data.shape)
        signal_with_noise = data + noise
        return np.asarray(signal_with_noise, dtype=np.int16)
