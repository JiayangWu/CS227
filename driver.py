from train_and_test import trainAndTest
from dataset_lists import small_datasets

# dataset = "UMD"
# EPOCHS = 500
# enable_data_augmentation = False
# percentage_similarity_loss = 0
# LSTM = False
for percentage_similarity_loss in [0]:
    for dataset in small_datasets:
        try:
                trainAndTest(dataset, percentage_similarity_loss = percentage_similarity_loss, enable_data_augmentation = True)
        except:
                with open("./logs.txt", "a") as f:
                        f.write(",".join([dataset, str(percentage_similarity_loss)]) + "\n")
        