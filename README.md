# Readme

## Repo Structures
- `auto_encoder.py`: My model
- `utilities.py`: Untilities functions
- `train_and_test.py`: The main function to train and test a model
- `dataset_lists.py`: Stores the name of all small datasets
- `kshape.py`: For SBD calculation
- `single_testing.py`: For testing under one setting
- `batched_testing.py`: For testing under batch
- `logs.txt`: Stores failed testing log, during failed testing the loss can be Nan and the output images are empty.
- `Experiments2.ipynb`: The notebook for early testing. It's no longer in use.
- `dataset_info`: Stores the informaiton of datasets
- `SBD_results`: Stores batched testing results using SBD as distance measure
- `L2_results`: Stores batched testing results using L2 as distance measure

## How to use
1. Download and unpack the file [here](https://drive.google.com/file/d/13PwgJNBTnyT1IjbUxFqQlqq2VTGDVw8N/view?usp=sharing) to the `data/` directory
2. To run a sample evaluation:
   open `single_testing.py` and set up the configurations, then call trainAndTest().

## Updates so far:
0. Similarity loss
1. Data Augmentation(both random and same, only random one was kept)
2. Pre-calculated SBD
3. Architectures without BN
4. Architectures with LSTM
5. Batched testing of SBD
6. Batched testing of ED

## Bugs fixed so far:
1. Too few epochs, previously 50, now 300
2. Used np library to calculate ED, causing No Gradients Provided for Any Variable and the model is not trained with similarity loss, now use TensorFLow
3. Updated the 10-nn score function, previously it used ED in the input, now it's properly the distance measure that I selected. 
4. Only z-normlized the training data, now testing data would be z-normalized as well
