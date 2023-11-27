import torch
import datasets
import pandas as pd
import numpy as np
import random
import math
import os
from torch.utils.data import DataLoader
from sentence_transformers import InputExample


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

class CovidDataLoader:
    def __init__(self) -> None:
        self.dataset = None

    def load_data(self, trec_covid_csv_path):
      if os.path.exists(trec_covid_csv_path):
          self.dataset = pd.read_csv(trec_covid_csv_path, index_col="doc_query_key")
      else:
          trec_covid_documents = pd.DataFrame(datasets.load_dataset("BeIR/trec-covid", "corpus", cache_dir='./data/cache')["corpus"])
          trec_covid_queries = pd.DataFrame(datasets.load_dataset("BeIR/trec-covid-generated-queries", "train", cache_dir='./data/cache')["train"])
          trec_covid_qrels = pd.DataFrame(datasets.load_dataset("BeIR/trec-covid-qrels", "test", cache_dir='./data/cache')["test"])

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
          self.dataset['doc'] = self.dataset.apply(lambda row: row['doc_title'] + ' ' + row['doc'] 
                          if pd.notnull(row['doc_title']) and pd.notnull(row['doc']) 
                          else row['doc_title'] if pd.notnull(row['doc_title']) 
                          else row['doc'], axis=1)
          self.dataset.dropna(subset=['doc', 'query', 'qrel_score'])
          self.dataset = self.dataset[self.dataset['doc'] != '']
          self.dataset = self.dataset[self.dataset['query'] != '']
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
    
    def make_dataloader(self, df: pd.DataFrame, shuffle=True, batch_size=16):
        input_examples = []
        for query, doc, label in zip(df["query"], df["doc"], df["qrel_score"]):
            # input_examples.append(InputExample(texts=[query, doc]))
            input_examples.append(InputExample(texts=[query, doc], label=label))
        return DataLoader(input_examples, shuffle=shuffle, batch_size=batch_size)
