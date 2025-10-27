import jieba
import jieba.analyse as analyse

text = '在实践中，当给定相同的查询、键和值的集合时， 我们希望模型可以基于相同的注意力机制学习到不同的行为， 然后将不同的行为作为知识组合起来， 捕获序列内各种范围的依赖关系 （例如，短距离依赖和长距离依赖关系）。 因此，允许注意力机制组合使用查询、键和值的不同 子空间表示（representation subspaces）可能是有益的。'
seg_list = jieba.cut(text)# 精确模式
print("Default Mode: " + "/ ".join(seg_list))
analyse.extract_tags(text,topK=5)
print("Top 5 keywords: " + ", ".join(analyse.extract_tags(text,topK=5)))
