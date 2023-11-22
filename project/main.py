import torch
from models.model_handler import Model
from data_loading.data_loader import CovidDataLoader, DataObj
from torch.utils.data import DataLoader, TensorDataset

bert = Model()

dataPreprocessor = CovidDataLoader()
dataPreprocessor.load_data()
train_set, val_set, test_set = dataPreprocessor.split_data(train=0.6, val=0.2, test=0.2)

print(len(dataPreprocessor.dataset))
print(len(train_set))
print(len(test_set))
print(len(val_set))
print(
    f"lost rows 😦 ): {len(dataPreprocessor.dataset) - len(train_set) - len(test_set) - len(val_set)}"
)

# tokenized_queries = bert.tokenizer.tokenize(train_set["query"])
# tokenized_docs = bert.tokenizer.tokenize(train_set["doc"])
query_emb = bert.model.bert.encode(list(train_set["query"]))
doc_emb = bert.model.bert.encode(list(train_set["doc"]))
query_tensor = torch.tensor(query_emb)
doc_tensor = torch.tensor(doc_emb)
labels_tensor = torch.tensor(train_set["qrel_score"])

# Create DataLoader
dataset = TensorDataset(query_tensor, doc_tensor, labels_tensor)
batch_size = 8
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# train_dataloader = DataLoader(train_set, batch_size=8, shuffle=True)
# val_dataloader = DataLoader(val_set, batch_size=8, shuffle=True)
# test_dataloader = DataLoader(test_set, batch_size=8, shuffle=True)

bert.train(train_dataloader=train_dataloader, learning_rate=2e-5, num_epochs=1)
# bert.train(train_dataloader=train_dataloader, learning_rate=2e-5, num_epochs=1, val_dataloader=val_dataloader)
# bert.evaluate(val_dataloader)
# bert.predict_examples()
