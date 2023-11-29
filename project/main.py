import os
import torch
import random
import numpy as np
from models.model_handler import ModelHandler
from data_loading.data_loader import CovidDataLoader
from models.peft_bert import PEFTBERTClassifier
from models.bert import BERTClassifier


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

bert = ModelHandler(BERTClassifier, run_name="frozen_bert_test")
peft_bert = ModelHandler(PEFTBERTClassifier, run_name="peft_bert_test")

data_dir = os.path.join(os.getcwd(), "project", "data")
# trec_covid_csv_path = os.path.join(data_dir, "trec_covid_beir.csv")
trec_covid_csv_path = os.path.join(data_dir, "dummy_data.csv")

dataPreprocessor = CovidDataLoader()
dataPreprocessor.load_data(trec_covid_csv_path=trec_covid_csv_path)
train_set, val_set, test_set = dataPreprocessor.split_data(train=0.6, val=0.2, test=0.2)
train_dataloader = dataPreprocessor.make_dataloader(train_set, shuffle=True, batch_size=32) 
val_dataloader = dataPreprocessor.make_dataloader(val_set, shuffle=True, batch_size=32) 

print(len(dataPreprocessor.dataset))
print(len(train_set))
print(len(test_set))
print(len(val_set))
print(
    f"lost rows 😦 ): {len(dataPreprocessor.dataset) - len(train_set) - len(test_set) - len(val_set)}"
)

print("Training peft model")
peft_bert.train(
    train_dataloader=train_dataloader, 
    # train_evaluation_steps=100, 
    # val_evaluation_steps=100, 
    learning_rate=1e-4, 
    num_epochs=4, 
    val_dataloader=val_dataloader
)

print("Training bert model")
bert.train(
    train_dataloader=train_dataloader, 
    evaluation_steps=100, 
    learning_rate=1e-4, 
    num_epochs=4, 
    val_dataloader=val_dataloader
)
# bert.evaluate(val_dataloader)
# bert.predict_examples()
