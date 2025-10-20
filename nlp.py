import jieba.posseg as psg

text = ('您好,请稍等^-^')
# 词性标注
seg = psg.cut(text)
for st, label in enumerate(seg):
    print(st, label)
