import re
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig, logging, tokenization_utils_base
import numpy as np
from evaluation import compute_f1
from os.path import exists

logging.set_verbosity_error()
torch.manual_seed(20000401)
np.random.seed(20000401)



def read_dataset(path="./NLP-2-Dataset/train", have_ans=True, df_saved=True):
    def calc_bm25_score(row): # row: [articles, question]
        seqs = re.findall(r"<s>\W*(.*?)\W*<\/s>",row[0]) # split articles by <s> and </s> tags
        tokenized_article = [seq.split(" ") for seq in seqs]
        tokenized_question = row[1].split(" ")
        score = BM25Okapi(tokenized_article).get_top_n(tokenized_question,seqs,n=5)
        return score

    def find_index(row): # row: [article, answer]
        all_pos = [[m.start(),m.end()] for m in re.finditer(re.escape(row[1]),row[0])] or [[0,0]]
        return all_pos

    if df_saved and exists(path+".json"):
        return pd.read_json(path+".json")
    
    cols = ["article", "question", "answer"] if have_ans else ["article", "question"]
    df = pd.read_csv(path+".txt", header=None, sep=r" \|\|\| ", names=cols, engine="python", encoding="utf-8", na_filter=False)
    df = df.dropna(axis=1)
    tqdm.pandas(desc="Calc bm25 score", mininterval=1)
    df["article"] = df[["article","question"]].progress_apply(calc_bm25_score ,axis=1) # select top 5 articles
    df = df.explode("article").reset_index(drop=True) # flatten list of articles
    tqdm.pandas(desc="Calc answer index", mininterval=1)
    df["answer_index"]= df[["article", "answer"]].progress_apply(find_index, axis=1) # find answer index in each article
    if df_saved:
        df.to_json(path+".json", indent=4)
    return df

class QADataSet(Dataset):
    def __init__(self, df:pd.DataFrame):
        assert all([col in df.columns for col in ["article", "question", "answer", "answer_index"]])
        self.df = df

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx:int):
        return self.df.iloc[idx].to_list()


class QAModel(nn.Module):
    def __init__(self, model_name="deepset/roberta-base-squad2"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name,config=self.config)
        for param in self.model.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, x):
        out = self.model(**x) # (batch_size, token_size(max512), 768)
        out = self.linear(out.last_hidden_state)# (batch_size, token_size, 2)
        out[:,:,0] = out[:,:,0].softmax(dim=-1) # (batch_size, token_size, 2) start
        out[:,:,1] = out[:,:,1].softmax(dim=-1) # (batch_size, token_size, 2) end
        return out

class QALoss(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.batch_size = batch_size

    def forward(self, batch_pred_pos, batch_match_pos, c2t_fn):
        batch_match_token_pos = self.get_char_to_token(batch_match_pos, c2t_fn)
        batch_loss = []
        for match_pos, pred_pos in zip(batch_match_token_pos, batch_pred_pos): # loop in batch
            loss = torch.stack([ self.start_and_end_loss(pos, pred_pos) for pos in match_pos ]) # multiple answer
            batch_loss.append(loss.min()) # choose mini loss in multiple answer
        final_loss = torch.stack(batch_loss).mean() # mean the batch loss
        return final_loss
    
    def start_and_end_loss(self,target_pos, pred_pos):
        # loss = start loss + end loss
        return self.loss_fn(pred_pos[:,0], target_pos[0]) + self.loss_fn(pred_pos[:,1], target_pos[1])

    def get_char_to_token(self, batch_match_pos, c2t_fn):
        batch_match_token_pos = []
        for i, match_pos in enumerate(batch_match_pos):
            if match_pos == [[0,0]]:
                batch_match_token_pos.append(torch.tensor(match_pos))
            else:
                batch_match_token_pos.append(torch.tensor([[c2t_fn(i, pos[0], 1), c2t_fn(i, pos[1]-1, 1)] for pos in match_pos])) # [start, end]
        return batch_match_token_pos



# tokenizer.encode() = CLS + tokenizer.tokenize().convert_tokens_to_ids() + SEP
# tokenizer() == tokenizer.encode() + support info (mask) 
# print(model.tokenizer.tokenize(batch_questions[0]))
# print(model.tokenizer.convert_tokens_to_ids(model.tokenizer.tokenize(batch_questions[0])))
# print(model.tokenizer.encode(batch_questions[0])) # 0=CLS(<s>), 2=SEP(</s>)
# print(model.tokenizer.decode(model.tokenizer.encode(batch_questions[0]))) # recombine split word, contains CLS and SEP

class Trainer():
    def __init__(self, batch_size=8):
        self.model = QAModel()
        self.qa_loss = QALoss(batch_size)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.save_interval = 15000
        self.val_interval = 10000

    def train(self):
        self.model.train()
        self.init_dataloader("train")
        p_bar = tqdm(self.train_dataloader, mininterval=1, desc="Training Batch, loss=0.0000", leave=False)
        
        for i, (batch_articles, batch_questions, _, batch_match_pos) in enumerate(p_bar):
            token = self.model.tokenizer(batch_questions, batch_articles, return_tensors="pt", padding=True, truncation=True)
            batch_pred_pos = self.model(token) # out put token index
            loss = self.qa_loss(batch_pred_pos, batch_match_pos, token.char_to_token)
            p_bar.set_description(f"Training, loss={loss:.4f}")
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if i+1 % self.save_interval == 0:
                torch.save(self.model.state_dict(), "./models/model_final.pt")
            if i+1 % self.val_interval == 0:
                self.validate()

    def validate(self):
        self.model.eval()
        self.init_dataloader("val")
        f1_scores = []
        with torch.no_grad():
            p_bar = tqdm(self.val_dataloader, mininterval=1, desc="Validation Batch, loss=0.0000")
            for batch_articles, batch_questions, batch_answer, batch_match_pos in p_bar:
            # for batch_articles, batch_questions, batch_answer, batch_match_pos in self.val_dataloader:
                token = self.model.tokenizer(batch_questions, batch_articles, return_tensors="pt", padding=True, truncation=True)
                batch_pred_pos = self.model(token)
                loss = self.qa_loss(batch_pred_pos, batch_match_pos, token.char_to_token)
                # p_bar.set_description("Testing, loss={:.4f}".format(loss.item()))
                batch_pred = self.pred_to_seq_index(batch_pred_pos, token) # TODO: fix None type error
                # f1_scores.append(compute_f1())
                # print(batch_pred, batch_answer[0])
        self.model.train()

    def pred_to_seq_index(self, batch_pred_pos, token:tokenization_utils_base.BatchEncoding):
        batch_pred = [] # [[s,e],[s,e]]
        for i in range(self.batch_size):
            start_token_index = batch_pred_pos[i,:,0].argmax(dim=-1).item()
            start_word_index = token.token_to_word(i,start_token_index)
            end_token_index = batch_pred_pos[i,:,1].argmax(dim=-1).item()
            end_word_index = token.token_to_word(i,end_token_index)
            if start_word_index is None and end_word_index is None:
                s,e = token.word_to_chars(i,0,1) # choose first word
            elif start_word_index is None or end_word_index is None:
                idx = start_word_index if end_word_index is None else end_word_index
                s,e = token.word_to_chars(i, idx,1) # choose start or end word index
            else:
                s = token.word_to_chars(i, start_word_index,1)[0] # choose start word start index
                e = token.word_to_chars(i, end_word_index,1)[1] # choose end word end index
            batch_pred.append([s, e])
        return batch_pred

    def init_dataloader(self, mode):
        if mode == "train" and not hasattr(self, "train_dataloader"):
            self.train_dataloader = DataLoader(QADataSet(read_dataset()), batch_size=self.batch_size, shuffle=True, collate_fn=lambda x:zip(*x))
        elif mode == "val" and not hasattr(self, "val_dataloader"):
            self.val_dataloader = DataLoader(QADataSet(read_dataset("./NLP-2-Dataset/val")), batch_size=self.batch_size, shuffle=False, collate_fn=lambda x:zip(*x))
        elif mode == "test" and not hasattr(self, "test_dataloader"):
            self.test_dataloader = DataLoader(QADataSet(read_dataset("./NLP-2-Dataset/test")), batch_size=self.batch_size, shuffle=False, collate_fn=lambda x:zip(*x))

    def load_from_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))) # TODO: device change

    def save_checkpoint(self, path=None):
        if path is None:
            path = f"./models/model_{int(pd.Timestamp.now().timestamp())}.pt"
        torch.save(self.model.state_dict(), path)
        


trainer = Trainer()
trainer.train()
# trainer.load_from_checkpoint("./models/model_final.pt")

