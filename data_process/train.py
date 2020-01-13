from model import *
from gensim.models.word2vec import Word2Vec
import pandas as pd
ENC_EMD_DIM=256
DEC_EMD_DIM=256
ENC_HID_DIM=512
DEC_HID_DIM=512
ENC_DROPOUT=0.7
OUTPUT_DROPOUT=0.7



if __name__ =="__mian__":
    root = 'data/'
    train_data = pd.read_pickle(root+'train_block.pkl')
    val_data = pd.read_pickle(root + 'dev_block.pkl')
    test_data = pd.read_pickle(root+'test_block.pkl')

    identifier_word2vec = Word2Vec.load(root+"identifier_w2v_128").wv
    token_word2vec = Word2Vec.load(root+"token_w2v_128").wv

    identifier_embeddings = np.zeros((identifier_word2vec.syn0.shape[0] + 1, identifier_word2vec.syn0.shape[1]), dtype="float32")
    identifier_word2vec[:identifier_word2vec.syn0.shape[0]] = identifier_word2vec.syn0