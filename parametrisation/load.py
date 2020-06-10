import scipy.io.wavfile
from data_extraction.speech_metadata import voice_actors


class LoadFiles:

    def load_file(self, filenames):
        signals = {
            'jackson': [],
            'nicolas': [],
            'theo': [],
            'yweweler': []
        }
        for actor in voice_actors:
            for filename in filenames[actor]:
                sample_rate, signal = scipy.io.wavfile.read(filename)
                signals[actor].append(signal)
        return signals, sample_rate
