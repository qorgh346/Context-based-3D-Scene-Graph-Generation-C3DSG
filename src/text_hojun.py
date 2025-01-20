
from gensim.scripts.glove2word2vec import glove2word2vec,KeyedVectors
import os
import numpy as np
import sys

import pickle
import gzip

words = []
idx = 0
word2idx = {}
glove_path = '/home/baebro/hojun_ws/3D_SGTR/'

#
word2vector = {}
# glove_input_file = '/home/baebro/hojun_ws/3D_SGTR/glove.840B.300d.txt'
glove_input_file = '/home/baebro/glove.6B.50d.txt'

word2vec_output_file = '/home/baebro/hojun_ws/3DSSG_Baek/glove.6B.5d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)

model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

word_vec = ['open','close','placing','gripping','other','visible','invisible']
pkl_keys = dict()
for word in word_vec:
    pkl_keys[word] = model[word]

with open('tempWordVec.pickle','wb') as f:
    pickle.dump(pkl_keys,f)


sys.exit()


with open(os.path.join(glove_input_file)) as file:

    for line in file:
        list_of_values = line.split()
        word = list_of_values[0]

        # vector_of_word = np.asarray(list_of_values[1:],dtype='float32')
        # print(list_of_values[1:])
        try:
            vector_of_word = np.asarray(list_of_values[1:],dtype='float32')
        except:
            pass
        # print(vector_of_word)
        # break
        word2vector[word] = vector_of_word

msg = f"Total number of words and corresponding vectors in word2vectors are {len(word2vector)}"
print(msg)
for i,v in word2vector.items():
    print(i)
