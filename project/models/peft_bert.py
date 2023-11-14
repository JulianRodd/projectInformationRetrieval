from torch import nn
from transformers import LoraConfig, LoraForSequenceClassification

class PEFTBERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(PEFTBERTClassifier, self).__init__()
        lora_config = LoraConfig.from_pretrained(bert_model_name, lora_alpha=16, lora_r=8)
        self.bert = LoraForSequenceClassification.from_pretrained(bert_model_name, config=lora_config, num_labels=num_classes)

        # Freeze
        for name, param in self.bert.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits