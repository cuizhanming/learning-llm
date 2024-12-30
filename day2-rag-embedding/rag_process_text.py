"""
# !pip install pdfminer.six
# !pip install nltk
"""
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import jieba
import nltk

nltk.download('stopwords')

"""
Step 1: Process Files, Ready for Embedding the data
Extracting paragraphs from PDF file:
page_numbers: list of integers, optional. If specified, only extract text from the specified pages.
min_line_length: int, optional. Minimum length of a line to be considered as a paragraph.
"""
def chunking_text_by_paragraphs(filename, page_numbers=None, min_line_length=1):
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    return paragraphs

# paragraphs = extract_paragraphs_from_pdf("llama2.pdf", min_line_length=10)
#
# for para in paragraphs[:4]:
#     print(para+"\n")

def chunking_text_by_size(paragraphs, chunk_size=300, overlap_size=100):
    """
    Split the text into chunks with a given size and overlap.
    """
    # sent_tokenize is for English, for Chinese see the `sent_tokenize_chinese` method
    sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        overlap = ''
        prev = i - 1

        # overlap parts from previous chunk
        while prev >= 0 and len(sentences[prev])+len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' ' + overlap
            prev -= 1
            prev_len = len(overlap)

        chunk = overlap + chunk
        next = i + 1

        # chunk for the next chunk_size
        while next < len(sentences) and len(sentences[next])+len(chunk) <= chunk_size:
            chunk += ' ' + sentences[next]
            next += 1
        chunks.append(chunk)
        i = next
    return chunks

# paragraphs = extract_paragraphs_from_pdf("llama2.pdf", min_line_length=10)
# chunks = chunking_text(paragraphs, chunk_size=300, overlap_size=100)
# print(chunks[:])

def to_keywords_chinese(input_string):
    """将句子转成检索关键词序列"""
    # 按搜索引擎模式分词
    word_tokens = jieba.cut_for_search(input_string)
    # 加载停用词表
    stop_words = set(stopwords.words('chinese'))
    # 去除停用词
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

def sent_tokenize_chinese(input_string):
    """按标点断句"""
    # 按标点切分
    sentences = re.split(r'(?<=[。！？；?!])', input_string)
    # 去掉空字符串
    return [sentence for sentence in sentences if sentence.strip()]

# # 测试关键词提取
# print(to_keywords_chinese("小明硕士毕业于中国科学院计算所，后在麻省理工学院深造"))
# # 测试断句
# print(sent_tokenize_chinese("这是，第一句。这是第二句吗？是的！啊"))


