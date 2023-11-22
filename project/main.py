from models.model_handler import Model
from data_loading.test_data_loader_imdb import load_imdb_data
from torch.utils.data import DataLoader
from models.peft_bert import PEFTBERTClassifier
from models.bert import BERTClassifier

bert = Model(PEFTBERTClassifier, device = "mps", run_name="peftbert")

train_dataset, val_dataset = load_imdb_data(bert.max_length, bert.tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

bert.train(train_dataloader=train_dataloader, learning_rate=2e-5, num_epochs=1, val_dataloader=val_dataloader)
bert.evaluate(val_dataloader)
bert.predict_examples()
