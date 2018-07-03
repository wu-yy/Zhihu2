#encoding:utf-8
'''
将问题 生成train.npz
'''
import numpy as np
import tqdm
import tensorflow as tf
def main(question_file,outfile, b_=50,c_=30,d_=250,e_=120):
    print("****begin****")

    with open(question_file ) as f:
        lines = f.readlines()

    results= [0 for _ in range(len(lines))]

    char2id = np.load('char_embedding.npz')['word2id'].item()
    word2id = np.load('word_embedding.npz')['word2id'].item()
    # print("char2id:\n",char2id)
    # print("word2id:\n",word2id)
    char_keys = set(char2id.keys())
    word_keys = set(word2id.keys())
    # print("char_keys:\n",char_keys)
    # print("word_keys:\n",word_keys)
    line=lines[0]

    print("line:\n",line)

    def process(line):
        a, b, c, d, e = line.replace('\n', '').split('\t')
        #print("a:\n",a)
        #print("b:\n",b)
        b, c, d, e = [_.split(',') for _ in [b, c, d, e]]
        b, c, d, e = [[int(word[1:]) if word != '' else -1 for word in sen] for sen in [b, c, d, e]]
        #print("b:\n",b)
        return b, c, d, e, a
    for ii, line in tqdm.tqdm(enumerate(lines)):
         results[ii] = process(line)
         #print(results[ii])
    del lines
    pad_sequence = tf.contrib.keras.preprocessing.sequence.pad_sequences
    bs = [[char2id['c'+str(_) if 'c'+str(_)  in char_keys else '</s>'] for _ in line[0]] for line in results]
    #print(bs[0])
    b_len = np.array([len(_) for _ in bs])
    bs_packed = pad_sequence(bs,maxlen=b_,padding='pre',truncating='pre',value = 0)
    print('a')
    del bs

    cs =  [[word2id['w'+str(_) if 'w'+str(_)  in word_keys else '</s>'] for _ in line[1]] for line in results]
    c_len = np.array([len(_) for _ in cs])
    cs_packed = pad_sequence(cs,maxlen=c_,padding='pre',truncating='pre',value = 0)
    print('b')
    del cs

    ds =  [[char2id['c'+str(_) if 'c'+str(_)  in char_keys else '</s>'] for _ in line[2]] for line in results]
    d_len = np.array([len(_) for _ in ds])
    ds_packed = pad_sequence(ds,maxlen=d_,padding='pre',truncating='pre',value = 0)
    print('c')
    del ds

    es =  [[word2id['w'+str(_) if 'w'+str(_)  in word_keys else '</s>'] for _ in line[3]] for line in results]
    e_len =  np.array([len(_) for _ in es])
    es_packed = pad_sequence(es,maxlen=e_,padding='pre',truncating='pre',value = 0)
    print('d')
    del es

    qids = [_[4] for _ in results]
    index2qid = {ii:jj for ii,jj in enumerate(qids)}

    np.savez_compressed(outfile,title_char = bs_packed,title_word = cs_packed,content_char = ds_packed,content_word = es_packed,title_char_len = b_len,title_word_len = c_len, content_char_len = d_len, content_word_len = e_len,index2qid = index2qid)



if __name__=='__main__':
    import fire
    fire.Fire()

