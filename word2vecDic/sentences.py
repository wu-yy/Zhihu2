##使用genism的word2vec
import os
#import gensim
class Mysentences(object):

    def __init__(self,dirname):
        self.dirname=dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname,fname)):
                yield list(map(lambda x:x.lower(),line.split()))

a="ge".lower()
sentences=Mysentences('data')
for i in sentences:
    print(i)
print(sentences)
#model=gensim.models.Word2Vec(sentences)
