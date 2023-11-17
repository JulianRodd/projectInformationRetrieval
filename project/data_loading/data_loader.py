import os

import pandas as pd
import torch
import datasets

class DataLoader(torch.utils.data.Dataset):
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

def load_data(max_length, tokenizer):
    data_dir = os.path.join(os.getcwd(), "project", "data")
    trec_covid_csv_path = os.path.join(data_dir, "trec_covid_beir.csv")
    
    if os.path.exists(trec_covid_csv_path):
        trec_covid = pd.read_csv(trec_covid_csv_path, index_col="doc_query_key")
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
        
        trec_covid = pd.merge(
            trec_covid_query_document_pairs,
            trec_covid_qrels,
            left_on="_id",
            right_on="corpus-id",
            how="inner"
        )
        
        trec_covid.drop("corpus-id", axis=1, inplace=True)
        trec_covid.index.name = "doc_query_key"
        trec_covid.rename(columns={
            "_id": "doc_id",
            "title": "doc_title",
            "text": "doc",
            "query-id": "query_id",
            "score": "qrel_score"
        }, inplace=True)
        
        trec_covid.to_csv(trec_covid_csv_path)
        
