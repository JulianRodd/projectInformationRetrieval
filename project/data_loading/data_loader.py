import pandas as pd
import numpy as np
from datasets import load_dataset, concatenate_datasets


class DataLoader:
    def __init__(self) -> None:
        self.dataset = None

    def load_data(self):
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
        print(self.dataset.describe())
        self.dataset.to_csv("dataset.csv")


dataPreprocessor = DataLoader()
dataPreprocessor.load_data()

print(dataPreprocessor.dataset)

"""
"""
