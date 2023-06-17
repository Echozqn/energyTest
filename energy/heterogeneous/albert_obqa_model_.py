import copy
import time
import numpy as np
# import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer, AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForSeq2SeqLM, AutoModel
)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# 记得去学下Automodels，AutoConfig，AutoTokenizer
class Moudle(nn.Module):
    """
    Base Transformer model that uses Pytorch Lightning as a PyTorch wrapper.

    T5 specific methods are implemented in T5Trainer
    """

    def __init__(self, config):
        super(Moudle, self).__init__()
        self.config = config
        self.choice_num = config.choice_num
        self.Device = config.device
        self.model = AutoModelForMaskedLM.from_pretrained(config.PLM)
        self.tokenizer = config.tokenizer
        self.bert = self.model.albert
        self.score = self.model.predictions
        self.mask_token = self.tokenizer("[MASK]")['input_ids'][1]
        
        self.hidden_size = self.model.config.hidden_size
        self.bn = nn.BatchNorm1d(self.hidden_size, affine=False).to(config.device)
        self.dropout = nn.Dropout(config.dropout)

        self.classifier = nn.Linear(self.hidden_size, 1, bias=True)
        nn.init.normal_(self.classifier.weight, std=self.model.config.initializer_range)
        self.classifier.bias.data.zero_()


    def forward_cls(self, batch, val=True):
        choice_num = self.choice_num
        batch_size = batch["input_ids"].size(0)
        batch["input_ids"] = batch["input_ids"].reshape(batch_size * choice_num, -1)
        batch["attention_mask"] = batch["attention_mask"].reshape(batch_size * choice_num, -1)
        batch["token_type_ids"] = batch["token_type_ids"].reshape(batch_size * choice_num, -1)

        outputs_real = self.bert(input_ids=batch["input_ids"].to(self.Device),
                                 attention_mask=batch["attention_mask"].to(self.Device),
                                 token_type_ids=batch["token_type_ids"].to(self.Device),
                                 output_hidden_states=True)

        real_emb = outputs_real.hidden_states[-1]
        embeddings = real_emb[:,0,:]

        logits = self.classifier(self.dropout(embeddings))
        # print(logits.shape)
        logits = logits.reshape(batch_size, choice_num)

        labels = torch.Tensor([0 for i in range(batch_size)]).long()
        
        from torch.nn import CrossEntropyLoss
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels.to(self.Device))
        return loss

    def training_step(self, batch, batch_idx=None):
        return self.forward_cls(batch, val=False)

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        optimizer = AdamW(self.parameters(), lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        self.opt = optimizer
        return optimizer

    def optimizer_step(self, optimizer):
        # print(optimizer)
        optimizer.step()
        optimizer.zero_grad()

    def cal_dev(self, batch):
        loss2, logits, z1, z2, batch_size = self.forward_cls(batch, val=True)
        logits = nn.functional.softmax(logits, dim=-1).cpu()
        #print(logits.shape, batch_size)
        ans = 0
        for i in range(batch_size):
            if logits[i][0] >= max(logits[i,1:self.choice_num].tolist()):
                ans += 1
        return ans, batch_size