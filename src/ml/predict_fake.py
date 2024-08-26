import os
import gc
import sys
import cv2
import math
import time
import tqdm
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator

from transformers import (AutoModel,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          AutoConfig,
                          get_cosine_schedule_with_warmup,
                          T5Tokenizer

                         )

from colorama import Fore, Back, Style





r_ = Fore.RED
b_ = Fore.BLUE
c_ = Fore.CYAN
g_ = Fore.GREEN
y_ = Fore.YELLOW
m_ = Fore.MAGENTA
sr_ = Style.RESET_ALL

config = {
#     'lr': 2e-5,
    'lr': 0.00002,
#     'wd':0.01,
    'wd':1e-5,
    'batch_size':16,
    'valid_step':50,
    'max_len':512,
    'epochs':8,
    'nfolds':5, # もう少し小さくてもよいかも
    'seed':42,
     'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
#    'device': xm.xla_device(),
    
#     https://huggingface.co/cl-tohoku
    'model_name':'cl-tohoku/bert-base-japanese'
#     'model_name':'cl-tohoku/bert-base-japanese-v2'
#     'model_name':'cl-tohoku/bert-large-japanese'
#     'model_name':''
}


# Configuration

# データセットの定義
class SeqDataset(Dataset):
    def __init__(self,df,tokenizer,max_len=128):
        self.targets = df['isfake'].to_numpy()
        self.context = df['context'].to_numpy()
        self.max_len = max_len
        self.tokenizer = tokenizer
    
    def __getitem__(self,idx):
        encode = self.tokenizer(self.context[idx],
                                return_tensors='pt',
                                max_length=self.max_len,
                                padding='max_length',
                                truncation=True)
 
        target = torch.tensor(self.targets[idx],dtype=torch.float) 
        return encode, target
    
    def __len__(self):
        return len(self.context)

class AttentionHead(nn.Module):
    ''' 
    BERT のヘッドにつけるアテンション機構。デフォルトのものを用いても良いが、オリジナルのものを作成して学習することも可能
    '''
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector
    

class Model(nn.Module):
    '''
    モデル本体
    '''
    def __init__(self,path):
        super(Model,self).__init__()
        self.roberta = AutoModel.from_pretrained(path)  
        self.config = AutoConfig.from_pretrained(path)

        self.head = AttentionHead(self.config.hidden_size,self.config.hidden_size)
#         self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.config.hidden_size,1)

    def forward(self,**xb):
        x = self.roberta(**xb)[0]
        x = self.head(x)
#         x = self.dropout(x)
        x = self.linear(x)
        return x    
class TestSeqDataset(Dataset):
    def __init__(self,df,tokenizer):
        self.excerpt = df['context'].to_numpy()
        self.tokenizer = tokenizer
    
    def __getitem__(self,idx):
        encode = self.tokenizer(self.excerpt[idx],return_tensors='pt',
                                max_length=config['max_len'],
                                padding='max_length',truncation=True)
        return encode
    
    def __len__(self):
        return len(self.excerpt)

def get_prediction(df,path,device=config['device']):        
    model = Model(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model.load_state_dict(torch.load(path,map_location=device))
    model.to(device)
    model.eval()
    
    test_ds = TestSeqDataset(df,tokenizer)
    test_dl = DataLoader(test_ds,
                        batch_size = config["batch_size"],
                        shuffle=False,
                        num_workers = 4,
                        pin_memory=True)
    
    predictions = list()
    for i, (inputs) in tqdm(enumerate(test_dl)):
        inputs = {key:val.reshape(val.shape[0],-1).to(device) for key,val in inputs.items()}
        outputs = model(**inputs)
        outputs = outputs.cpu().detach().numpy().ravel().tolist()
        predictions.extend(outputs)
        
    torch.cuda.empty_cache()
    return np.array(predictions)

# 評価指標
def eval_fn(outputs,targets):
    outputs =  torch.tensor(outputs, dtype=torch.float) 
    targets =  torch.tensor(targets, dtype=torch.float) 
    outputs = outputs.view(-1)
    targets = targets.view(-1)
    return torch.sqrt(nn.MSELoss()(outputs,targets)).cpu().detach().numpy().ravel().tolist()[0]



def load_models_and_run_inference( input_, config):
    """
    Load models and tokenizers, then run inference on the input.

    Parameters:
    - model_paths: List of file paths to the model files.
    - input_: The input data for inference.
    - config: A dictionary containing model configuration, including 'model_name' and 'device'.

    Returns:
    - y_pred: The predicted label for the input.
    """

    # Initialize lists to store models and tokenizers
    model_paths=model_paths = ['../../models/model0/model0.bin', '../../models/model1/model1.bin', '../../models/model2/model2.bin', '../../models/model3/model3.bin']

    models = []
    tokenizers = []

    # Load models and tokenizers
    for model_path in model_paths:
        model = Model(config['model_name'])
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        state_dict = torch.load(model_path, map_location=config['device'])
        model.load_state_dict(state_dict, strict=False)
        models.append(model)
        tokenizers.append(tokenizer)

    # Prepare input data for inference
    reference_df = pd.DataFrame({"context": [input_]})

    preds = []

    # Run inference using each model and tokenizer
    for model, tokenizer in zip(models, tokenizers):
        pred = get_prediction(reference_df, model, tokenizer)
        preds.append(pred)

    # Average predictions across all models
    pred = np.mean(preds, axis=0)

    # Map predictions to labels
    def map_to_labels(pred, thresholds=(0.25, 1.5)):
        pred = np.array(pred)
        labels = np.zeros_like(pred, dtype=int)
        labels[pred > thresholds[0]] = 1
        labels[pred > thresholds[1]] = 2
        return labels.ravel()

    y_pred = map_to_labels(pred)[0]

    # Output the prediction
    if y_pred == 0:
        print("This is original real news")
    elif y_pred == 1:
        print("This is fake news, but is half fake")
    else:
        print("This is fake news, and is fully fake")

    return y_pred