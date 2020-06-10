"""
using https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""

import numpy
from scipy.fftpack import dct

import matplotlib.pyplot as plt

class MFCC:

    __pre_emphasis = 0.97
    __frame_size = 0.025  # [ms]
    __frame_stride = 0.015  # [ms] in other words it is frame length
    __NFFT = 512  # fast fourier transform's length
    __num_of_filters = 40  # number of mel filters
    __cep_lifter = 22
    __target_energy = 10000  # signal is adjusted to have this max peak

    def __init__(self, num_mfcc, file_length):
        self.num_mfcc = num_mfcc
        self.file_length = file_length  # [s] (depending on training number value - 0.75 to 2.4)


    def calculate_mfcc(self, fs, signal):
        signal = self.__zero_padding(signal, fs)
        signal = self.__adjust_energy(signal)
        emphasized_signal = self.__pre_emphasise(signal)
        frames = self.__framing(emphasized_signal, fs)
        frame_spectrum = self.__fourier_transform(frames)
        filter_banks = self.__filter_banks(frame_spectrum, fs)
        return self.__mfcc(filter_banks)


    def __adjust_energy(self, signal):
        max_ = numpy.max(signal)
        scalar = self.__target_energy / max_
        return signal * scalar


    def __zero_padding(self, signal, fs):
        """For data to have the same length. Later it's decorelated, so it's not a problem"""
        signal_target_length = int(self.file_length * fs)
        signal_actual_length = len(signal)
        try:
            return numpy.pad(signal, int((signal_target_length-signal_actual_length)/2), 'constant', constant_values=(0))
        except ValueError:
            raise Exception('You have to change file_length, because it is too small.')


    def __pre_emphasise(self, signal):
         return numpy.append(signal[0], signal[1:] - self.__pre_emphasis * signal[:-1])


    def __framing(self, signal, sample_rate):
        frame_length, frame_step = self.__frame_size * sample_rate, self.__frame_stride * sample_rate
        signal_length = len(signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = numpy.zeros((pad_signal_length - signal_length))
        pad_signal = numpy.append(signal, z)

        indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
            numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(numpy.int32, copy=False)]

        frames *= numpy.hamming(frame_length)  # hamming window

        return frames


    def __fourier_transform(self, frames):
        mag_frames = numpy.absolute(numpy.fft.rfft(frames, self.__NFFT))  # Magnitude of the FFT
        return (1.0 / self.__NFFT) * (mag_frames ** 2)  # Power Spectrum


    def __filter_banks(self, frames, sample_rate):
        low_freq_mel = 0
        high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, self.__num_of_filters + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = numpy.floor((self.__NFFT + 1) * hz_points / sample_rate)

        fbank = numpy.zeros((self.__num_of_filters, int(numpy.floor(self.__NFFT / 2 + 1))))
        for m in range(1, self.__num_of_filters + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = numpy.dot(frames, fbank.T)
        filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * numpy.log10(filter_banks)  # dB

        return filter_banks


    def __mfcc(self, filter_banks):
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (self.num_mfcc + 1)]
        (n_frames, n_coeff) = mfcc.shape
        n = numpy.arange(n_coeff)
        lift = 1 + (self.__cep_lifter / 2) * numpy.sin(numpy.pi * n / self.__cep_lifter)
        mfcc *= lift  # *

        filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
        mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

        return mfcc
