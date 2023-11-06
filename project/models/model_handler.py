import torch
from tqdm import tqdm
from torch import nn
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from .bert import BERTClassifier


class Model:
	def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 3, max_length: int = 128, freeze = True) -> None:
		self.bert_model_name = model_name
		self.num_classes = num_classes
		self.max_length = max_length

		self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = BERTClassifier(self.bert_model_name, num_classes).to(self.device)

		if freeze:
			#TODO check if this freezes the right layers
			for param in self.model.bert.parameters():
				param.requires_grad = False


	def train(self, train_dataloader: DataLoader, learning_rate: float = 2e-5, num_epochs: int = 4, val_dataloader = DataLoader|None):
		optimizer = AdamW(self.model.parameters(), lr=learning_rate)
		total_steps = len(train_dataloader) * num_epochs
		scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

		for epoch in range(num_epochs):
			print(f"Epoch {epoch + 1}/{num_epochs}")
			self.run_epoch(train_dataloader, optimizer, scheduler)
			if val_dataloader is not None:
				accuracy, report = self.evaluate(val_dataloader)
				print(f"Validation Accuracy: {accuracy:.4f}")
				print(report)

		torch.save(self.model.state_dict(), "bert_classifier.pth")

	def run_epoch(self, data_loader, optimizer, scheduler):
		self.model.train()
		for batch in tqdm(data_loader):
			optimizer.zero_grad()
			input_ids = batch['input_ids'].to(self.device)
			attention_mask = batch['attention_mask'].to(self.device)
			labels = batch['label'].to(self.device)
			outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
			loss = nn.CrossEntropyLoss()(outputs, labels)
			loss.backward()
			optimizer.step()
			scheduler.step()

	def evaluate(self, data_loader):
		self.model.eval()
		predictions = []
		actual_labels = []
		with torch.no_grad():
			for batch in tqdm(data_loader):
				input_ids = batch['input_ids'].to(self.device)
				attention_mask = batch['attention_mask'].to(self.device)
				labels = batch['label'].to(self.device)
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
				_, preds = torch.max(outputs, dim=1)
				predictions.extend(preds.cpu().tolist())
				actual_labels.extend(labels.cpu().tolist())
		return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

	def predict_sentiment(self, text):
		self.model.eval()
		encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
		input_ids = encoding['input_ids'].to(self.device)
		attention_mask = encoding['attention_mask'].to(self.device)

		with torch.no_grad():
			outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
			_, preds = torch.max(outputs, dim=1)
		return "positive" if preds.item() == 1 else "negative"

	def predict_examples(self):
		# Test sentiment prediction
		test_text = "The movie was great and I really enjoyed the performances of the actors."
		sentiment = self.predict_sentiment(test_text)
		print("The movie was great and I really enjoyed the performances of the actors.")
		print(f"Predicted sentiment: {sentiment}")

		# Test sentiment prediction
		test_text = "The movie was so bad and I would not recommend it to anyone."
		sentiment = self.predict_sentiment(test_text)
		print("The movie was so bad and I would not recommend it to anyone.")
		print(f"Predicted sentiment: {sentiment}")

		# Test sentiment prediction
		test_text = "Worst movie of the year."
		sentiment = self.predict_sentiment(test_text)
		print("Worst movie of the year.")
		print(f"Predicted sentiment: {sentiment}")
