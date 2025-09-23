from collections import defaultdict, Counter
import jieba


def tokenize(sentence):
    return [char for char in sentence]


def count_ngrams(corpus, n=2):
    ngrams_count = defaultdict(Counter)
    for sentence in corpus:
        tokens = tokenize(sentence)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i: i + n])
            prefix = ngram[:-1]
            token = ngram[-1]
            ngrams_count[prefix][token] += 1
    return ngrams_count


# 构建一个玩具数据集
corpus = ["我喜欢吃苹果，而且我喜欢吃香蕉",
          "他不喜欢吃香蕉，但他喜欢吃苹果",
          "她喜欢吃草莓，我也喜欢吃草莓"]

seg_list1 = jieba.cut(corpus[0])  # 精准分离出语料库中的句子
seg_list2 = jieba.cut(corpus[0], cut_all=True)  # 全模式分离出语料库中的句子
print('/'.join(seg_list1))
print('/'.join(seg_list2))
# 对每个句子进行分词，并打印出对应的单字列表
print("单字列表:")
tokens = tokenize(corpus)
print(tokens)

bigram_counts = count_ngrams(corpus, n=2)
print("bigram 词频：")
for prefix, counts in bigram_counts.items():
    print(f'{''.join(prefix)}: {dict(counts)}')
