import torch
from torch import nn
from sentence_transformers import SentenceTransformer


#TODO tensorboard support. Inherit from SentenceTransformer and change SentenceTransformer_eval_during_training
class BERTClassifier(nn.Module):
    def __init__(
            self, 
            bert_model_name, 
            num_classes, 
            concatenation_args: dict = {
                "concatenation_sent_rep": True,
                "concatenation_sent_difference": True,
                "concatenation_sent_multiplication": False,
            }
    ):
        super(BERTClassifier, self).__init__()
        self.bert = SentenceTransformer(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = self.bert[0].auto_model.base_model.config.hidden_size
        self.fc = nn.Linear(self.hidden_size, num_classes)
        self.concatenation_args: dict = concatenation_args
        num_vectors_concatenated = 0
        if concatenation_args["concatenation_sent_rep"]:
            num_vectors_concatenated += 2
        if concatenation_args["concatenation_sent_difference"]:
            num_vectors_concatenated += 1
        if concatenation_args["concatenation_sent_multiplication"]:
            num_vectors_concatenated += 1
        self.sent_embed_linear = nn.Linear(num_vectors_concatenated * self.hidden_size, num_classes)

    def forward(self, *args, **kwargs):
        labels_in_kwargs = "labels" in kwargs
        labels = kwargs.pop("labels", None)
        if labels_in_kwargs:
            outputs, logits = self.get_sentence_embedding(*args, **kwargs)
            return outputs, logits
        else:
            outputs = self.bert(*args, **kwargs)
            sentence_embedding = outputs["sentence_embedding"]
            x = self.dropout(sentence_embedding)
            logits = self.fc(x)
            return outputs

    def get_sentence_embedding(self, *args, **kwargs):
        outputs = [self.bert(inputs) for inputs in args[0]]
        rep_a, rep_b = [rep['sentence_embedding'] for rep in outputs]

        vectors_concat = []
        if self.concatenation_args["concatenation_sent_rep"]:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_args["concatenation_sent_difference"]:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_args["concatenation_sent_multiplication"]:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        logits = self.sent_embed_linear(features)
        #TODO check if this output shape is equal to that of the normal self.bert output
        return outputs, logits