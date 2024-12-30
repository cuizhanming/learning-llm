from transformers import BertModel
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

pretrained = (BertModel
              .from_pretrained(
    r"../local_models/google-bert/bert-base-chinese/models--google-bert--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
              .to(DEVICE))
print(pretrained)

# 定义下游任务模型（增量模型）
class MyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 设计全连接层， 实现二分类任务
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 冻结Bert模型的参数，让其不参与训练
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            print(out)
        out = self.fc(out.last_hidden_state[:,0])
        return out

"""
Import Dataset
"""
from torch.utils.data import Dataset
from datasets import load_from_disk

class MyDataset(Dataset):

    def __init__(self, subset="test"):
        # 加载数据集
        self.dataset = load_from_disk(r"../local_datasets/ChnSentiCorp")
        if subset == "train":
            self.dataset = self.dataset["train"]
        elif subset == "test":
            self.dataset = self.dataset["test"]
        elif subset == "validation":
            self.dataset = self.dataset["validation"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, row):
        text = self.dataset[row]["text"]
        label = self.dataset[row]["label"]
        return text, label

test_dataset = MyDataset("test")
for data in test_dataset:
    print(data)

"""
Training
"""
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer,AdamW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# times
EPOCH = 30000

tokenizer=BertTokenizer.from_pretrained(
    r"../local_models/google-bert/bert-base-chinese/models--google-bert--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
train_dataset = MyDataset("train")
def collate_fn(batch):
    texts, labels = zip(*batch)
    # Tokenize the texts
    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=texts,
        max_length=500,
        padding="max_length",
        return_tensors="pt",
        return_length=True,
        truncation=True)

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]

    labels = torch.tensor(labels)
    return input_ids, attention_mask, token_type_ids, labels

train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True,
    # Drop last set, to avoid Error! "RuntimeError: DataLoader worker (pid(s) 1234) exited unexpectedly"
    drop_last=True,
    # Encodes the inputs dataset
    collate_fn= collate_fn
)

def run_training():
    print(DEVICE)
    # 定义模型
    model = MyModel().to(DEVICE)
    # 定义优化器
    optimize_model = AdamW(model.parameters(), lr=1e-5)
    # 定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for i,(input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # 数据加载到设备
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            # labels = labels.to(DEVICE)
            # 梯度清零
            optimize_model.zero_grad()
            # 前向传播
            outputs = model(input_ids, attention_mask, token_type_ids)
            # 计算损失
            loss = loss_func(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimize_model.step()
            if i % 5 == 0:
                outputs = outputs.argmax(dim=1) # Get the predicted labels
                out_accuracy_rate = (outputs == labels).sum().item() / len(labels)
                print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}, Accuracy: {out_accuracy_rate}")

        # Save the model
        torch.save(model.state_dict(), f"params/bert_{epoch}.pth")

"""

"""

names=["negative","positive"]
model = Model().to(DEVICE)

def collate_fn2(text):
    # Tokenize the texts
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=text,
        max_length=500,
        padding="max_length",
        return_tensors="pt",
        return_length=True,
        truncation=True)

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]

    return input_ids, attention_mask, token_type_ids

def test():
    model.load_state_dict(torch.load("params/bert_0.pth", map_location=DEVICE))
    model.eval()

    while True:
        data = input("Please input the text:(exit when input `q`):")
        if data == "q":
            print("Exit!")
            break

        input_ids,attention_mask,token_type_ids = collate_fn2(data)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids)
            outputs = outputs.argmax(dim=1)
            print(outputs)
            print(f"Model output: {names[outputs.item()]}")



if __name__ == "__main__":
    # Training
    run_training()
    # Testing
    test()