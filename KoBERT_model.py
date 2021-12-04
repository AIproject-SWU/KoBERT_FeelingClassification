import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook, notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from streaming_gstt import main
import sys

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"

##GPU 사용 시
device = torch.device("cpu")

bertmodel, vocab = get_pytorch_kobert_model()

# 새로운 데이터셋 불러오는 코드
import pandas as pd
dataset_train1 = pd.read_csv('C:\\Users\\llsa0\\PBL4_speechtext\\with-sy\\model\\data\\data.csv')

# 감정 라벨링
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(dataset_train1['Emotion'])
dataset_train1['Emotion'] = encoder.transform(dataset_train1['Emotion'])

# 라벨링된 감정 매핑 {0: '공포', 1: '놀람', 2: '분노', 3: '슬픔', 4: '중립', 5: '행복', 6: '혐오'}
mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))
mapping

# # Train / Test set 분리
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset_train1, test_size=0.2, random_state=42)
print("train shape is:", len(train))
print("test shape is:", len(test))

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# 테스트데이터셋 전처리
class BERTDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i]) for i in dataset['Sentence']]
        self.labels = [np.int32(i) for i in dataset['Emotion']]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# 실시간 입력 형식을 위한 데이터변환기
class BERTDataset2(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

## Setting parameters
max_len = 64
batch_size = 16
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

data_train = BERTDataset(train,  tok, max_len, True, False)
data_test = BERTDataset(test, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
# AdamW
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
# SGD
#optimizer = optim.SGD(optimizer_grouped_parameters, lr=0.01, momentum=0.9)

loss_fn = nn.CrossEntropyLoss()
t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

#모델 state_dict 불러오기
#모델 초기화
model1 = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
model1.eval()
PATH = 'C:\\Users\\llsa0\\PBL4_speechtext\\with-sy\\model\\data\\'
checkpoint = torch.load(PATH + 'model_10_2.tar')
model1.load_state_dict(checkpoint['model_state_dict'], strict=False)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 해당 인덱스값이 어떤 감정을 나타내는지 바꿔줌.
def switch_feel(feel_num):
    if feel_num == 0:
        return "공포"
    elif feel_num == 1:
        return "놀람"
    elif feel_num == 2:
        return "분노"
    elif feel_num == 3:
        return "슬픔"
    elif feel_num == 4:
        return "중립"
    elif feel_num == 5:
        return "행복"
    elif feel_num == 6:
        return "혐오"
    return "무감정"


# 실시간 입력형 테스트방식
def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset2(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

    model1.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model1(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            best = np.argmax(logits)
            test_eval.append(best)
            # 가장 높은것의 값을 -10으로 바꾸어 그 다음으로 높은값을 argmax로 쉽게 찾을 수 있게 함.
            logits[best] = -10
            test_eval.append(np.argmax(logits))

        # 두번째로 높은 값이 0보다 작을경우에는 입력값의 감정과 연관이 없다고 판단되어 결과출력에서 제외함.
        if logits[test_eval[1]] < 0:
            sys.stdout.write(GREEN)
            print(">> 입력내용에서 가장 크게 느껴지는 감정은 " + switch_feel(test_eval[0]) + " 입니다.")
        else:
            sys.stdout.write(GREEN)
            print(">> 입력내용에서 가장 크게 느껴지는 감정은 " + switch_feel(test_eval[0]) + " 이고, 두번째는 " + switch_feel(
                test_eval[1]) + " 입니다.")

if __name__ == "__main__":

    #입출력 반복 실행. 0 입력시 종료
    end = 1
    while end == 1 :
        main()
        f = open('test.txt', 'rt', encoding='UTF-8')
        sentence = f.readline()
        predict(sentence[:-2])
        file = open("test.txt", "w", encoding='UTF-8')
        print("\n")