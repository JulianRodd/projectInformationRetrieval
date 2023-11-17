import torch
import datasets
import pandas as pd
import numpy as np
import random
import math
import os
from datasets import load_dataset, concatenate_datasets


class DataObj(torch.utils.data.Dataset):
    def __init__(self, texts, labels, max_length, tokenizer):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, 
            return_tensors='pt', 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(), 
            'attention_mask': encoding['attention_mask'].flatten(), 
            'label': torch.tensor(label)
        }

class DataLoader:
    def __init__(self) -> None:
        self.dataset = None

    def load_data(self):
      data_dir = os.path.join(os.getcwd(), "project", "data")
      trec_covid_csv_path = os.path.join(data_dir, "trec_covid_beir.csv")

      if os.path.exists(trec_covid_csv_path):
          self.dataset = pd.read_csv(trec_covid_csv_path, index_col="doc_query_key")
      else:
          trec_covid_documents = pd.DataFrame(datasets.load_dataset("BeIR/trec-covid", "corpus")["corpus"])
          trec_covid_queries = pd.DataFrame(datasets.load_dataset("BeIR/trec-covid-generated-queries", "train")["train"])
          trec_covid_qrels = pd.DataFrame(datasets.load_dataset("BeIR/trec-covid-qrels", "test")["test"])

          trec_covid_query_document_pairs = pd.merge(
              trec_covid_queries,
              trec_covid_documents,
              on=["_id", "text", "title"],
              how="inner"
          )

          self.dataset = pd.merge(
              trec_covid_query_document_pairs,
              trec_covid_qrels,
              left_on="_id",
              right_on="corpus-id",
              how="inner"
          )

          self.dataset.drop("corpus-id", axis=1, inplace=True)
          self.dataset.index.name = "doc_query_key"
          self.dataset.rename(columns={
              "_id": "doc_id",
              "title": "doc_title",
              "text": "doc",
              "query-id": "query_id",
              "score": "qrel_score"
          }, inplace=True)

          self.dataset.to_csv(trec_covid_csv_path)

    def split_data(self, train, val, test):
        random.seed(42)
        querie_id_set = list(self.dataset["query_id"].unique())
        train_interval_queries = [0, math.floor(len(querie_id_set)*train)]
        val_interval_queries = [math.floor(len(querie_id_set)*train), math.floor(len(querie_id_set)*(val+train))]
        test_interval_queries = [math.floor(len(querie_id_set)*(val+train)), 0]
        random.shuffle(querie_id_set)
        train_queries = querie_id_set[:train_interval_queries[1]]
        val_queries = querie_id_set[val_interval_queries[0]:val_interval_queries[1]]
        test_queries = querie_id_set[test_interval_queries[0]:]
        
        text_set = list(self.dataset["doc_id"].unique())
        train_interval_texts = [0, math.floor(len(text_set)*train)]
        val_interval_texts = [math.floor(len(text_set)*train), math.floor(len(text_set)*(val+train))]
        test_interval_texts = [math.floor(len(text_set)*(val+train)), 0]
        random.shuffle(text_set)
        train_texts = text_set[:train_interval_texts[1]]
        val_texts = text_set[val_interval_texts[0]:val_interval_texts[1]]
        test_texts = text_set[test_interval_texts[0]:]

        train_set = self.dataset[self.dataset['query_id'].isin(train_queries)][self.dataset['doc_id'].isin(train_texts)]
        val_set = self.dataset[self.dataset['query_id'].isin(val_queries)][self.dataset['doc_id'].isin(val_texts)]
        test_set = self.dataset[self.dataset['query_id'].isin(test_queries)][self.dataset['doc_id'].isin(test_texts)]
        return train_set, val_set, test_set

      
def check_unique(df_1, df_2):
    queries_1 = set(df_1["query-id"])
    queries_2 = set(df_2["query-id"])
    texts_1 = set(df_1["text_y"])
    texts_2 = set(df_2["text_y"])
    query_intersect = queries_1.intersection(queries_2)
    text_intersect = texts_1.intersection(texts_2)
    print(len(query_intersect))
    print(len(text_intersect))
    

dataPreprocessor = DataLoader()
dataPreprocessor.load_data()
train_set, val_set, test_set = dataPreprocessor.split_data(train=0.6, val=0.2, test=0.2)
print(f"Len train set: {len(train_set)}")
print(train_set.head())
print(f"Len val set: {len(val_set)}")
print(val_set.head())
print(f"Len test set: {len(test_set)}")
print(test_set.head())

check_unique(train_set, val_set)
check_unique(val_set, test_set)
check_unique(train_set, test_set)
