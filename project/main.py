import os
import torch
from models.model_handler import Model
from data_loading.data_loader import CovidDataLoader, DataObj
from torch.utils.data import DataLoader, TensorDataset
from models.peft_bert import PEFTBERTClassifier
from models.bert import BERTClassifier
from sentence_transformers import InputExample


bert = Model(BERTClassifier, run_name="frozen_bert")
peft_bert = Model(PEFTBERTClassifier, run_name="peft_bert")

data_dir = os.path.join(os.getcwd(), "project", "data")
trec_covid_csv_path = os.path.join(data_dir, "trec_covid_beir.csv")
# trec_covid_csv_path = os.path.join(data_dir, "dummy_data.csv")

dataPreprocessor = CovidDataLoader()
dataPreprocessor.load_data(trec_covid_csv_path=trec_covid_csv_path)
train_set, val_set, test_set = dataPreprocessor.split_data(train=0.6, val=0.2, test=0.2)
train_dataloader = dataPreprocessor.make_dataloader(train_set, shuffle=True, batch_size=16) 
val_dataloader = dataPreprocessor.make_dataloader(val_set, shuffle=True, batch_size=16) 

print(len(dataPreprocessor.dataset))
print(len(train_set))
print(len(test_set))
print(len(val_set))
print(
    f"lost rows 😦 ): {len(dataPreprocessor.dataset) - len(train_set) - len(test_set) - len(val_set)}"
)

# tokenized_queries_train = bert.tokenizer.batch_encode_plus(list(train_set["query"]))
# tokenized_queries_train = bert.tokenizer.batch_encode_plus(list(train_set["doc"]))
# tokenized_docs = bert.tokenizer.tokenize(train_set["doc"])

# query_emb = bert.model.bert.encode(list(train_set["query"]))
# doc_emb = bert.model.bert.encode(list(train_set["doc"]))
# query_tensor = torch.tensor(query_emb)
# doc_tensor = torch.tensor(doc_emb)
# labels_tensor = torch.tensor(train_set["qrel_score"])

# bert.train(train_dataloader=train_dataloader, learning_rate=2e-5, num_epochs=1)
bert.train(train_dataloader=train_dataloader, learning_rate=2e-5, num_epochs=1, val_dataloader=val_dataloader)
bert.evaluate(val_dataloader)
bert.predict_examples()
