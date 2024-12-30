import torch
from mydata import MyDataset
from torch.utils.data import DataLoader
from mymodel import Model
from transformers import BertTokenizer, AdamW

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义训练轮次
epoch = 2

# 加载分词器
token = BertTokenizer.from_pretrained(r"D:\AIProject\huggingface\model_cache\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

# 自定义函数，对数据进行编码处理
def collate_fn(data):
    sentences = [i[0] for i in data]
    label = [i[1] for i in data]

    # 编码处理
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sentences,
        truncation=True,
        padding="max_length",
        max_length=350,
        return_tensors="pt",
        return_length=True
    )

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]

    labels = torch.LongTensor(label)

    return input_ids, attention_mask, token_type_ids, labels


# 创建数据集
test_dataset = MyDataset("test")

# 创建数据加载器 DataLoader
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=50,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

if __name__ == '__main__':
    acc = 0
    total = 0

    # 开始测试
    print(device)
    model = Model().to(device)

    # 加载第二轮训练保存的模型
    model.load_state_dict(torch.load("./params/1bert.pt"))

    model.eval()  # 开启测试模式

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        # 将数据放到 device 上
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device), labels.to(device)

        # 执行前向计算得到输出
        out = model(input_ids, attention_mask, token_type_ids)

        out = out.argmax(dim=1)
        # 计算精度（准确率）
        acc += (out == labels).sum().item()

        total += len(labels)

    print(acc / total)