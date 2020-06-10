from playsound import playsound
import simpleaudio as sa


class AudioPlayer:

    def play_audio_from_file(self, file_name):
        playsound('recordings\\{}'.format(file_name))

    def play_signal(self, signal, fs):
        play_object = sa.play_buffer(signal, num_channels=1, bytes_per_sample=2, sample_rate=fs)
        play_object.wait_done()
