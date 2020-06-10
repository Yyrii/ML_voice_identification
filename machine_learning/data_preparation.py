from data_extraction.file_assignment import FileAssigner
from data_extraction.speech_metadata import voice_actors, voice_actors_arrays
from data_presentation.player import AudioPlayer
from parametrisation.filtration import Filter
from parametrisation.mfcc import MFCC
from parametrisation.load import LoadFiles


class PrepareData:

    def __init__(self, filter_kind, cutoffs, numbers_train, white_noise, file_length=0.75, num_mfcc=12):
        self.filter_kind = filter_kind
        self.numbers_train = numbers_train
        self.cutoffs = cutoffs
        self.white_noise_aplitude = white_noise
        self.file_length = file_length
        self.num_mfcc = num_mfcc


    def prepare_data_for_dataset(self):
        mfcced_signals = {
            'jackson': [],
            'nicolas': [],
            'theo': [],
            'yweweler': []
        }
        choose_filter = {
            'low': Filter().lowpass_filter,
            'band': Filter().bandpass_filter,
            'none': Filter().no_filter
        }
        assigned_files = FileAssigner().assign_files()
        voices = voice_actors_arrays.copy()
        for number in self.numbers_train:
            prepared_files = self.__sort_objects(assigned_files, number=number, voices=voices)

        sorted_signals, sample_rate = LoadFiles().load_file(prepared_files)
        for actor in voice_actors:
            for signal in sorted_signals[actor]:
                signal = Filter().white_noise(signal, self.white_noise_aplitude)
                signal = choose_filter[self.filter_kind](signal, self.cutoffs, sample_rate, 6)
                mfcc_ = MFCC(num_mfcc=self.num_mfcc, file_length=self.file_length).calculate_mfcc(sample_rate, signal)
                mfcced_signals[actor].append(mfcc_)
        return mfcced_signals, len(mfcced_signals[voice_actors[0]][0])


    def __sort_objects(self, assigned_files, number, voices):
        for actor in voice_actors:
            voices[actor].extend(assigned_files[actor][number])
        return voices
