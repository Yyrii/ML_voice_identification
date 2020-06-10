from data_extraction.speech_metadata import voice_files
import os


class FileAssigner:

    __recordings_dir = 'recordings\\'

    def assign_files(self):
        assign_files = voice_files.copy()
        for dir, _, files in os.walk(self.__recordings_dir):
            files = [file for file in files if not file.startswith('__')]
            assign_files = self.__assign_file_to_speaker_and_val(files, dir, assign_files)
        try:
            return assign_files
        except Exception as exc:
            raise exc

    @staticmethod
    def __assign_file_to_speaker_and_val(files, dir, assign_files):
        for file in files:
            filename, _ = file.split('.')
            number, name, _ = filename.split('_')
            assign_files[name][number].append(dir + file)
        return assign_files
