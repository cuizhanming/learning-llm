import torch
from mydata import MyDataset
from torch.utils.data import DataLoader
from mymodel import Model
from transformers import BertTokenizer, AdamW

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义训练轮次，建议20-50次
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
train_dataset = MyDataset("train")

# 创建数据加载器 DataLoader
# 使用 DataLoader 实现批量数据加载。
# DataLoader 自动处理数据的批处理和随机打乱，确保训练的高效性和数据的多样性。
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=50,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)

if __name__ == '__main__':
    # 开始训练
    print(device)
    model = Model().to(device)

    # 定义优化器
    # AdamW 是一种适用于 BERT 模型的优化器，结合了 Adam 和权重衰减的特点，能够有效地防止过拟合。
    optimizer = AdamW(model.parameters(), lr=5e-4)

    # 定义损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    # 开启训练模式
    model.train()

    for epoch in range(epoch):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # 将数据放到 device 上
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), labels.to(device)

            # 前向传播（forward pass），执行前向计算得到输出
            out = model(input_ids, attention_mask, token_type_ids)

            # 损失计算（loss calculation）
            loss = loss_func(out, labels)

            optimizer.zero_grad()

            # 反向传播（backwardpass）
            loss.backward()

            optimizer.step()

            if i % 5 == 0:
                out = out.argmax(dim=1)
                # 计算精度（准确率）
                acc = (out == labels).sum().item() / len(labels)
                # 通常训练过程中会跟踪损失值的变化，以判断模型的收敛情况。
                print(epoch, i, loss.item(), acc)

        # 保存模型
        torch.save(model.state_dict(), f"params/{epoch}bert.pt")
        print(epoch, "参数保存成功！")