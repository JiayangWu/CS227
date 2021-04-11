from train_and_test import trainAndTest
from dataset_lists import small_datasets

# dataset = "UMD"
# EPOCHS = 500
# enable_data_augmentation = False
# percentage_similarity_loss = 0
# LSTM = False

for dataset in ["GunPoint"]:
    for percentage_similarity_loss in [0]:
        trainAndTest(dataset, percentage_similarity_loss = percentage_similarity_loss)