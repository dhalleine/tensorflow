from collections import Counter
wordcount = Counter(open("words.txt"))
for item in  wordcount.items():
    print "%d %s" % (item[1], item[0])
