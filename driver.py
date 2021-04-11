from train_and_test import trainAndTest
from dataset_lists import small_datasets

# dataset = "UMD"
# EPOCHS = 500
# enable_data_augmentation = False
# percentage_similarity_loss = 0
# LSTM = False

for dataset in small_datasets:
    for percentage_similarity_loss in [0, 0.5, 1]:
        trainAndTest(dataset, percentage_similarity_loss = percentage_similarity_loss)