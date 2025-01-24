#定制化生成内容
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

model_path = r"../local_models/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

#加载我们自己训练的权重（中文古诗词）
trained_parameter_path = r"../local_params/net.pt"
loaded_object = torch.load(trained_parameter_path, map_location=torch.device('cpu'))
model.load_state_dict(loaded_object)

#定义函数，用于生成5言绝句 text是提示词，row是生成文本的行数，col是每行的字符数。
def generate(text, row, col):

    #定义一个内部递归函数，用于生成文本
    def generate_loop(data):
        #禁用梯度计算
        with torch.no_grad():
            #使用data字典中的数据作为模型输入，并获取输出
            out = model(**data)
        #获取最后一个字(logits未归一化的概率输出)
        out = out["logits"]
        #选择每个序列的最后一个logits，对应于下一个词的预测
        out = out[:,-1]

        #找到概率排名前50的值，以此为分界线，小于该值的全部舍去
        topk_value = torch.topk(out,50).values
        #获取每个输出序列中前50个最大的logits（为保持原维度不变，需要对结果增加一个维度，因为索引操作会降维）
        topk_value = topk_value[:,-1].unsqueeze(dim=1)
        #将所有小于第50大的值的logits设置为负无穷，减少低概率词的选择
        out = out.masked_fill(out < topk_value,-float("inf"))

        #将特殊符号的logits值设置为负无穷，防止模型生成这些符号。
        for i in ",.()《《[]「」{}":
            out[:,tokenizer.get_vocab()[i]] = -float('inf')
        out[:,tokenizer.get_vocab()["[PAD]"]] = -float('inf')
        out[:,tokenizer.get_vocab()["[UNK]"]] = -float('inf')
        out[:,tokenizer.get_vocab()["[CLS]"]] = -float('inf')
        out[:,tokenizer.get_vocab()["[SEP]"]] = -float('inf')

        #根据概率采样，无放回，避免生成重复的内容
        out = out.softmax(dim=1)
        #从概率分布中进行采样，选择下一个词的ID
        out = out.multinomial(num_samples=1)

        #强值添加标点符号
        #计算当前生成的文本长度于预期的长度的比例
        c = data["input_ids"].shape[1] / (col+1)
        #如果当前的长度是预期长度的整数倍，则添加标点符号
        if c % 1 ==0:
            if c % 2 ==0:
                #在偶数位添加句号
                out[:,0] = tokenizer.get_vocab()["."]
            else:
                #在奇数位添加逗号
                out[:,0] = tokenizer.get_vocab()[","]
        #将生成的新词ID添加到输入序列的末尾
        data["input_ids"] = torch.cat([data["input_ids"],out],dim=1)
        #更新注意力掩码，标记所有有效位置
        data["attention_mask"] = torch.ones_like(data["input_ids"])
        #更新token的ID类型，通常在BERTm模型中使用，但是在GPT模型中是不用的
        data["token_type_ids"] = torch.ones_like(data["input_ids"])
        #更新标签，这里将输入ID复制到标签中，在语言生成模型中通常用与预测下一个词
        data["labels"] = data["input_ids"].clone()

        #检查生成的文本长度是否达到或超过指定的行数和列数
        if data["input_ids"].shape[1] >= row*col + row+1:
            #如果达到长度要求，则返回最终的data字典
            return data
        #如果长度未达到要求，递归调用generate_loop函数继续生成文本
        return generate_loop(data)

    #生成3首诗词
    #使用tokenizer对输入文本进行编码，并重复3次生成3个样本。
    data = tokenizer.batch_encode_plus([text] * 3, return_tensors="pt")
    #移除编码后的序列中的最后一个token(结束符号)
    data["input_ids"] = data["input_ids"][:,:-1]
    #创建一个与input_ids形状相同的全1张量，用于注意力掩码
    data["attention_mask"] = torch.ones_like(data["input_ids"])
    # 创建一个与input_ids形状相同的全0张量，用于token类型ID
    data["token_type_ids"] = torch.zeros_like(data["input_ids"])
    #复制input_ids到labels，用于模型的目标
    data['labels'] = data["input_ids"].clone()

    #调用generate_loop函数开始生成文本
    data = generate_loop(data)

    #遍历生成的3个样本
    for i in range(3):
        #打印输出样本索引和对应的解码后的文本
        print(i,tokenizer.decode(data["input_ids"][i]))

if __name__ == '__main__':
    generate("白",row=4,col=5)

#输出结果
# 0 [CLS] 白 头 江 海 已, 归 路 向 南 州. 白 首 身 方 健, 青 春 路 尚 优.
# 1 [CLS] 白 衣 侍 从 不, [UNK] 首 看 清 光. 夜 入 青 云 洞, 秋 从 朱 [UNK] 墙.
# 2 [CLS] 白 昼 一 眠 无, 无 眠 时 卧 眠. 未 闻 鸡 叫 声, 但 听 鸣 山 鸦.
# 调优后，剔除特殊符号，生成的古诗词更加通顺，且不会生成特殊符号。
# 0 [CLS] 白 鹤 飞 来 石, 高 龙 出 洞 天. 烟 霞 随 处 尽, 风 月 对 谁 边.
# 1 [CLS] 白 菊 相 看 一, 高 阳 独 倚 君. 诗 从 三 峡 尽, 琴 在 半 山 闻.
# 2 [CLS] 白 玉 华 枝 不, 青 玉 色 已 深. 虽 非 不 如 白, 不 见 桃 与 金.