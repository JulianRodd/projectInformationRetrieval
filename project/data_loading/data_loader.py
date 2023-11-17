import pandas as pd
import numpy as np
import random
import math
import os
from datasets import load_dataset, concatenate_datasets

class DataLoader:
    def __init__(self) -> None:
        self.dataset = None

    def load_data(self):
        data_dir = os.path.join(os.getcwd(), "project", "data")
        trec_covid_csv_path = os.path.join(data_dir, "dataset.csv")
        
        if os.path.exists(trec_covid_csv_path):
            self.dataset = pd.read_csv(trec_covid_csv_path)
        else:
            dataset_qrels = pd.DataFrame(load_dataset("BeIR/trec-covid-qrels", "test")['test'])
            dataset_text = pd.DataFrame(load_dataset("BeIR/trec-covid", "corpus")['corpus'])
            dataset_queries = pd.DataFrame(load_dataset("BeIR/trec-covid-generated-queries", "train")['train'])
            self.dataset = pd.merge(pd.merge(
                dataset_qrels,
                dataset_text,
                left_on="corpus-id",
                right_on="_id",
                how="inner",
            ), dataset_queries, left_on="corpus-id", right_on="_id", how="inner")
            self.dataset.to_csv(os.path.join(os.getcwd(), "project", "data") + "/dataset.csv")
        print(self.dataset.describe())
        
    def print_groups_count(self):
        # Count occurrences of each query-id
        query_id_counts = self.dataset['query-id'].value_counts()

        # Print each query-id group and its count
        for query_id, count in query_id_counts.items():
            print(f"Query ID {query_id} occurs {count} times")

    def split_data(self, train, val, test):
        random.seed(42)
        querie_id_set = list(self.dataset["query-id"].unique())
        train_interval_queries = [0, math.floor(len(querie_id_set)*train)]
        val_interval_queries = [math.floor(len(querie_id_set)*train), math.floor(len(querie_id_set)*(val+train))]
        test_interval_queries = [math.floor(len(querie_id_set)*(val+train)), 0]
        random.shuffle(querie_id_set)
        train_queries = querie_id_set[:train_interval_queries[1]]
        val_queries = querie_id_set[val_interval_queries[0]:val_interval_queries[1]]
        test_queries = querie_id_set[test_interval_queries[0]:]
        
        text_set = list(self.dataset["text_y"].unique())
        train_interval_texts = [0, math.floor(len(text_set)*train)]
        val_interval_texts = [math.floor(len(text_set)*train), math.floor(len(text_set)*(val+train))]
        test_interval_texts = [math.floor(len(text_set)*(val+train)), 0]
        random.shuffle(text_set)
        train_texts = text_set[:train_interval_texts[1]]
        val_texts = text_set[val_interval_texts[0]:val_interval_texts[1]]
        test_texts = text_set[test_interval_texts[0]:]

        train_set = self.dataset[self.dataset['query-id'].isin(train_queries)][self.dataset['text_y'].isin(train_texts)]
        val_set = self.dataset[self.dataset['query-id'].isin(val_queries)][self.dataset['text_y'].isin(val_texts)]
        test_set = self.dataset[self.dataset['query-id'].isin(test_queries)][self.dataset['text_y'].isin(test_texts)]
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
# dataPreprocessor.print_groups_count()
train_set, val_set, test_set = dataPreprocessor.split_data(train=0.6, val=0.2, test=0.2)
# print(dataPreprocessor.dataset)
print(f"Len train set: {len(train_set)}")
print(train_set.head())
print(f"Len val set: {len(val_set)}")
print(val_set.head())
print(f"Len test set: {len(test_set)}")
print(test_set.head())

check_unique(train_set, val_set)
check_unique(val_set, test_set)
check_unique(train_set, test_set)


"""
"""
