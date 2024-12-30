import torch
from mymodel import Model
from transformers import BertTokenizer

# 定义评价分类名称
names = ["负向评价", "正向评价"]
# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Model().to(device)

token = BertTokenizer.from_pretrained(r"D:\AIProject\huggingface\model_cache\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

def collate_fn(data):
    sentences = [data]

    # 编码处理
    data = token.batch_encode_plus(
        batch_text_or_text_pairs = sentences,
        truncation = True,
        padding = "max_length",
        max_length = 350,
        return_tensors = "pt",
        return_length = True
    )

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]

    return input_ids, attention_mask, token_type_ids

def test():
    model.load_state_dict(torch.load("./params/1bert.pt"))
    model.eval()
    while True:
        data = input("请输入测试数据（输入'q'推出）：")

        if data == "q":
            print("测试结束！")
            break
        
        input_ids, attention_mask, token_type_ids= collate_fn(data)
        # 将数据放到 device 上
        input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)

        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
            out = out.argmax(dim = 1)
            print("模型判定：", names[out], "\n")

if __name__ == '__main__':
    test() 