from collections import Counter
wordcount = Counter(open("words.txt").read().split())
for item in wordcount.items():
    print "%d %s" % (item[1], item[0])
