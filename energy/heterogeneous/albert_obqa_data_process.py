import json
import time

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset


class Data:
    def get(self, s):
        return " ".join(s.strip().split())

    def make(self, x):
        c = []
        A = ord('A')
        for i in range(10):
            s = "answer"+chr(A+i)
            if s in x:
                c.append(self.get(x[s]))
        answer = x["correct"]
        if answer == "A":
            pass
        else:
            #print(answer,ord(answer)-A)
            c[0], c[ord(answer)-A] = c[ord(answer)-A], c[0]
        if "context" in x:
            a = [self.get(x["context"]+x["question"]),c]
        else:
            a = [self.get(x["question"]), c]
        return a

    def getdata(self, pathContext):  # 获取某一个数据集的数据
        context = []
        for line in open("./data/" + pathContext + ".jsonl", encoding="utf-8"):
            f = json.loads(line)
            context.append(self.make(f))
        temp = []
        for i, line in enumerate(open("./data/" + pathContext + ".tgt", encoding="utf-8")):
            line = line[:-1]
            #context[i][0] = line
            temp.append(line)
        assert len(temp) == len(context)
        return context

    def getall(self, trainContext):
        train = self.getdata(trainContext)
       
        return train


class GetDataset:
    def getDataLoader(self, trainContext, batch_size, tokenizer, input_max_len, config):
        data = Data()
        train = data.getall(trainContext)
        trainDataset = self.create_data_loader(train, tokenizer, config, input_max_len, batch_size, True,drop_last=True)
        return trainDataset

    def create_data_loader(self, context, tokenizer, config, input_max_len, batch_size, shuffle=True,drop_last=False):
        ds = TAGDataset(
            context=context,
            tokenizer=tokenizer,
            config=config,
            input_max_len=input_max_len,
            istrain=drop_last
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,drop_last=drop_last)


class TAGDataset(Dataset):
    def __init__(self, context, tokenizer, config, input_max_len, istrain):
        self.context = context
        self.tokenizer = tokenizer
        self.config = config
        self.input_max_len = input_max_len
        self.len = len(self.context)
        self.inputs = []
        self.targets = []
        self.l = []
        self.r = []
        self.istrain = istrain
        self._build()

    def __getitem__(self, item):
        tokenized_inputs = self.inputs[item]
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for i, inp in enumerate(tokenized_inputs):
            # print(inp)
            input_ids.append(inp["input_ids"].squeeze().reshape(1, -1))
            attention_mask.append(inp["attention_mask"].squeeze().reshape(1, -1))
            token_type_ids.append(inp["token_type_ids"].squeeze().reshape(1, -1))
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "token_type_ids": token_type_ids}

    def __len__(self):
        return self.len

    def get_token_num(self, s):
        return len(self.tokenizer.tokenize(s))

    def _build(self):
        cnt = 0
        for i in range(len(self.context)):
            context = self.context[i]
            conversation = context[0]
            reals = []
            for answer in context[1]:
                raw = conversation + " " + answer
                inputs = self.tokenizer.encode_plus(
                    raw, max_length=self.input_max_len, add_special_tokens=True, return_tensors='pt',
                    padding='max_length', truncation=True, return_attention_mask=True, return_token_type_ids=True,
                )
                reals.append(inputs)
                if (self.config["PLM"].find("albert") != -1 and inputs["input_ids"][0][-1] != 0):
                    cnt += 1
            self.inputs.append(reals)
        self.len = len(self.inputs)
        print("?", cnt)