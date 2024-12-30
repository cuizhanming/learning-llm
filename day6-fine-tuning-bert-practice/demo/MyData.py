#自定义数据集
from torch.utils.data import Dataset
from datasets import load_from_disk

class MyDataset(Dataset):
    def __init__(self,split):
        #从磁盘加载数据
        self.dataset = load_from_disk(r"../../data/ChnSentiCorp")
        if split == 'train':
            self.dataset = self.dataset["train"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]['text']
        label = self.dataset[item]['label']

        return text,label

    def to_csv(self,path):
        self.dataset.to_csv(path_or_buf=path)

if __name__ == '__main__':
    dataset = MyDataset("test")
    print(dataset.to_csv(r"./data/ChnSentiCorp.csv"))
    # for data in dataset:
    #     print(data)