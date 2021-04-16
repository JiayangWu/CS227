from train_and_test import trainAndTest
from dataset_lists import small_datasets

# dataset = "UMD"
# EPOCHS = 500
# enable_data_augmentation = False
# percentage_similarity_loss = 0
# LSTM = False
# for percentage_similarity_loss in [0.5, 1]:
#     for dataset in small_datasets:
#         try:
#                 trainAndTest(dataset, percentage_similarity_loss = percentage_similarity_loss, enable_data_augmentation = True, EPOCHS = 300)
#         except:
#                 with open("./logs.txt", "a") as f:
#                         f.write(",".join([dataset, str(percentage_similarity_loss)]) + "\n")

# trainAndTest("GunPoint", EPOCHS = 300)
for percentage_similarity_loss in [0, 0.2, 0.5, 1]:
    # for LSTM in [False]:
    for dataset in small_datasets:
        try:
            trainAndTest(dataset, percentage_similarity_loss = percentage_similarity_loss, EPOCHS = 300, save_output = True)
        except:
            with open("./logs.txt", "a") as f:
                f.write(",".join([dataset, str(percentage_similarity_loss)]) + "\n")

for percentage_similarity_loss in [0, 0.5, 1]:
    # for LSTM in [False]:
    for dataset in small_datasets:
        try:
            trainAndTest(dataset, percentage_similarity_loss = percentage_similarity_loss, enable_data_augmentation = True, EPOCHS = 300, save_output = True)
        except:
            with open("./logs.txt", "a") as f:
                f.write(",".join([dataset, str(percentage_similarity_loss)]) + "\n")

for percentage_similarity_loss in [0, 0.5, 1]:
    # for LSTM in [False]:
    for dataset in small_datasets:
        try:
            trainAndTest(dataset, percentage_similarity_loss = percentage_similarity_loss, LSTM = LSTM, EPOCHS = 300, save_output = True)
        except:
            with open("./logs.txt", "a") as f:
                f.write(",".join([dataset, str(percentage_similarity_loss)]) + "\n")
        