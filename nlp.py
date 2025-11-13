import jieba.analyse as analyse

key_word_num = 5
text = '记者从国家文物局获悉，截至3月15日，19个省（区、市）180多家博物馆在做好疫情防控工作的前提下恢复对外开放，其中19家为一级博物馆。另外，沈阳故宫博物院、新四军江南指挥部纪念馆、金沙遗址博物馆等将于3月17日陆续恢复开放。随着疫情防控形势好转，各地博物馆、纪念馆等陆续恢复开放。记者从各恢复开放博物馆发布的公告获悉，各恢复开放博物馆对疫情防控期间参观观众在提前预约、测量体温等提出了明确要求，并提醒观众做好个人防护。2月27日，国家文物局发布《关于新冠肺炎疫情防控期间有序推进文博单位恢复开放和复工的指导意见》强调，有序恢复开放文物、博物馆单位，各文物、博物馆开放单位可采取网上实名预约、总量控制、分时分流、语音讲解、数字导览等措施，减少人员聚集。'
key_words = analyse.textrank(text, topK=key_word_num)
print("Top 5 keywords: " + ", ".join(key_words))
key_words = analyse.extract_tags(text, topK=key_word_num)
print("Top 5 keywords: " + ", ".join(key_words))
