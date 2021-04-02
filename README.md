# Readme

## How to use

1. Download and unpack the file [here](https://drive.google.com/file/d/13PwgJNBTnyT1IjbUxFqQlqq2VTGDVw8N/view?usp=sharing) to the `data/` directory
2. To run a sample evaluation: `python3 sample_evaluation.py -d GunPoint`


## Updates so far:

1. Data Augmentation(both random and same)
2. Pre-calculated SBD
3. Pre-randomly select pairs

## Bugs fixed so far:
1. too few epochs, previously 50, now 500
2. used np library to calculate ED, causing No Gradients Provided for Any Variable and the model is not trained with similarity loss, now use tf
3. update the 10-nn score function, previously it reshaped the input to (45, 2), which is the shape of Libras. That's why we had small 10-NN score like 0.038. Now it will reshape according to input shape.
