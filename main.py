from machine_learning.data_preparation import PrepareData
from machine_learning.creating_dataset import MaskedVoicesDataset
from machine_learning.ml_manager import Manager


if __name__ == '__main__':
    mfcc_collection, frame_number = PrepareData(filter_kind='none', cutoffs=300, number_train='6', white_noise=.0).\
        prepare_data_for_dataset()

    dataset = MaskedVoicesDataset(mfcc_collection=mfcc_collection, mfcc_length=frame_number, len_samples=200)

    acc = Manager(rep_num=10, tt_distribution=[180, 20], batch_size=5, epochs=3, lr=0.001).calculate_efficiency(dataset)

    print('accuracy: {} %'.format(round(acc, 3)))
