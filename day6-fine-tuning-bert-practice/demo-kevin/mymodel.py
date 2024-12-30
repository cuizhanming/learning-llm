from transformers import BertModel
import torch

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载本地预训练模型
# 加载 Hugging Face 默认缓存路径的模型，只需要模型名称即可
# pretrained = BertModel.from_pretrained("google-bert/bert-base-chinese").to(device)
# 加载指定路径的模型，需要给出模型文件的绝对路径
pretrained = BertModel.from_pretrained(r"D:\AIProject\huggingface\model_cache\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f").to(device)
print(pretrained)

'''
1. 定义下游任务模型
2. 将主干网络所提取的特征进行分类
'''
class Model(torch.nn.Module):

    # 模型结构设计
    def __init__(self):
        super().__init__()

        # 一个全连接层，2表示情感分类为二分类（正向、负向）
        self.fc = torch.nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 上游任务不参与训练，权重锁死，只参与前向传播，不参反向传播进行梯度更新
        with torch.no_grad():
            # 得到上游任务（主干网络）的输出
            # input_ids, attention_mask, token_type_ids 是传递给上游任务的参数
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # 下游任务参与训练
        out = self.fc(out.last_hidden_state[:,0])
        out = out.softmax(dim=1)
        
        return out
    