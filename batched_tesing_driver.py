from train_and_test import trainAndTest
from dataset_lists import small_datasets

# trainAndTest("GunPoint", EPOCHS = 300)
for percentage_similarity_loss in [-1]:
    # for LSTM in [False]:
    for dataset in small_datasets:
        try:
            trainAndTest(dataset, percentage_similarity_loss = percentage_similarity_loss,  EPOCHS = 300, save_output = False, NlogN = False)
        except:
            with open("./logs.txt", "a") as f:
                f.write(",".join([dataset, str(percentage_similarity_loss)]) + "\n")
