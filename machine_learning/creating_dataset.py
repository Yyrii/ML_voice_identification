from torch.utils.data import Dataset
import torch
from data_extraction.speech_metadata import voice_actors


class MaskedVoicesDataset(Dataset):

    def __init__(self, mfcc_collection, mfcc_length, len_samples):
        self.mfcc_collection = mfcc_collection
        self.mfcc_length = mfcc_length
        self.len_samples = len_samples  # 200
        self.targeted_samples = self.make_list(self.mfcc_collection)

    def __len__(self):
        return 4 * len(self.mfcc_collection[voice_actors[0]])

    def __getitem__(self, idx):
        voice_actor_to_tensor = {
            'jackson': 0,
            'nicolas': 1,
            'theo': 2,
            'yweweler': 3
        }
        target = torch.tensor(voice_actor_to_tensor[self.targeted_samples[idx][0]], dtype=torch.int16)
        target_mfcc = self.targeted_samples[idx][1]
        return (target, target_mfcc)

    def make_list(self, samples):
        targeted_samples = []
        for actor in voice_actors:
            for mfcc in samples[actor]:
                targeted_samples.append([actor, mfcc])
        return targeted_samples
