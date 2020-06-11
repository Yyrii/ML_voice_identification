from machine_learning.data_preparation import PrepareData
from machine_learning.creating_dataset import MaskedVoicesDataset
from machine_learning.ml_manager import Manager
import time


def main(numbers_to_train, filter_kind, cutoffs, w_noise_amp, file_length, rep_num, tt_distribution, batch_size,
         learning_rate, epochs, num_mfcc):
    mfcc_collection, frame_number = PrepareData(filter_kind=filter_kind, cutoffs=cutoffs,
                                                numbers_train=numbers_to_train, white_noise=w_noise_amp,
                                                file_length=file_length, num_mfcc=num_mfcc).prepare_data_for_dataset()
    dataset = MaskedVoicesDataset(mfcc_collection=mfcc_collection, mfcc_length=frame_number, len_samples=200)
    return Manager(rep_num=rep_num, tt_distribution=tt_distribution, batch_size=batch_size, epochs=epochs,
                   lr=learning_rate, num_to_train=len(numbers_to_train)).calculate_efficiency(dataset, num_mfcc * frame_number)


if __name__ == '__main__':

    numbers_to_train = ['0','1','2','3',]       # 0 - 9, as many as you want. Be aware of 'file_length' argument.
    filter_kind = 'none'                        # either 'none', 'low' or 'band
    cutoffs = None                              # ex 'low': 500, 'band: [100, 1000]
    white_noise_amp = 0.1                       # part of signal's max peak ( 0.1 equals 10% max amplitude)
    file_length = 2.3                           # [s] zero padding to this file length
    repetitions = 10                            # number determine how much training / tests will be performed
    tt_distribution = [70, 30]                  # [%] train test batch size distribution
    batch_size = 5                              # how many samples are provided for ml per tick
    learning_rate = 0.001                       # pytorch's learning rate
    epochs = 3                                  # how many times, pytorch will pass through whole data for loss min.
    num_mfcc = 12

    start_time = time.time()
    acc = main(numbers_to_train=numbers_to_train, filter_kind=filter_kind, cutoffs=cutoffs, w_noise_amp=white_noise_amp,
               file_length=file_length, rep_num=repetitions, tt_distribution=tt_distribution, batch_size=batch_size,
               learning_rate=learning_rate, epochs=epochs, num_mfcc=num_mfcc)
    elapsed_time = time.time() - start_time
    print('Accuracy acquired: {} %'.format(round(acc, 3)))
    print('elapsed time: ', elapsed_time)
