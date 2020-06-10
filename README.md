# Machine learning masked voice identification

Trying to make pytorch efficient masked voice identificator. Works as well as text-dependend and text-independent, however relatively
little data was used for training.

## How to use
In main function, there are plenty arguments - that might be change the way that program works. Also, if you want to change `torch` net 
architecture, you can do this in `machine_learning\net.py`.

#### Example Usage
1. Change `numbers_to_train` list, append it with any numbers from 0 to 9.
2. Increase `white_noise_amp` to check how that influence accuracy.
3. Play with any variable you want, and check it's influence.


## How to run
1. Download and unpack project in any directory you want.
2. You have to download recordings from https://github.com/Jakobovski/free-spoken-digit-dataset, and paste them into `recording\` directory.
3. For windows run: `run.bat`
