import pandas as pd
import os
from nltk import word_tokenize
import xml.etree.ElementTree as ET
import numpy as np


class Pipeline:
    def __init__(self, root, part, identifier_dim_size, token_dim_size, path, max_identifier_voc_size=100000, max_token_voc_size=100000):
        self.root = root
        self.part = part
        self.identifier_dim_size = identifier_dim_size
        self.token_dim_size = token_dim_size
        self.max_identifier_voc_size = max_identifier_voc_size
        self.max_token_voc_size = max_token_voc_size
        self.path=path

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file):

        part = self.part

        trees = pd.read_json(input_file)

        from prepare_data import get_ast_sequences

        def trans_to_sequence(ast):
            ast = ET.fromstring(ast)
            sequnece = []
            get_ast_sequences(ast, sequnece)
            return sequnece

        # code sequence
        code_corpus = trees['code'].apply(trans_to_sequence)
        code_corpus_str = [' '.join(code) for code in code_corpus]
        trees['code'] = pd.Series(code_corpus_str)

        # comment sequence
        comment_corpus = trees['comment'].apply(word_tokenize)
        comment_corpus_str = [' '.join(comment) for comment in comment_corpus]
        trees['comment'] = pd.Series(comment_corpus_str)

        # dump
        if not os.path.exists(os.path.join(self.root, part)):
            os.makedirs(os.path.join(self.root, part))
        trees.to_json(os.path.join(self.root, part, "programs_ns.json"))

        # embedding
        from gensim.models.word2vec import Word2Vec

        # code
        w2v_code = Word2Vec(code_corpus, size=self.identifier_dim_size,
                            workers=8, sg=1, min_count=3, window=5, max_final_vocab=self.max_identifier_voc_size)

        if not os.path.exists(os.path.join(self.root, part)):
            os.makedirs(os.path.join(self.root, part))

        w2v_code.save(os.path.join(
            self.root, part, 'identifier_w2v_'+str(self.identifier_dim_size)))

        # comment
        w2v_comment = Word2Vec(comment_corpus, size=self.token_dim_size,
                               workers=8, sg=1, min_count=3, window=5, max_final_vocab=self.max_token_voc_size)
        # add entity
        wv = w2v_comment.wv
        embedding_size = wv.syn0.shape[1]
        entities = ["<sos>", "<eos>", "<pad>"]
        weights = np.random.normal(size=(len(entities), embedding_size))
        wv.add(entities=entities, weights=weights, replace=False)

        w2v_comment.save(os.path.join(
            self.root, part, 'token_w2v_'+str(self.token_dim_size)))

    # generate block sequences with index representations
    def generate_block_seqs(self, data_path):
        from prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        def comment_to_index(comment):
            comment=word_tokenize(comment)
            comment.insert(0, '<sos>')
            comment.append('<eos>')
            return [vocab_comment[token].index if token in vocab_comment else token_max for token in comment]

        def trans2seq(code_xml):
            def tree_to_index(node):
                token = node.tag
                result = [
                    vocab_code[token].index if token in vocab_code else identifier_max]
                children = node.getchildren()
                for child in children:
                    result.append(tree_to_index(child))
                return result

            root = ET.fromstring(code_xml)
            blocks = []
            func(root, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        part = self.part

        # code word2vec vocabulary
        word2vec_code = Word2Vec.load(os.path.join(self.root, part, 'identifier_w2v_'+str(self.identifier_dim_size))
                                      ).wv
        vocab_code = word2vec_code.vocab
        identifier_max = word2vec_code.syn0.shape[0]

        # comment word2vec vocabulary
        word2vec_comment = Word2Vec.load(os.path.join(self.root, part, 'token_w2v_'+str(self.token_dim_size))
                                         ).wv
        vocab_comment = word2vec_comment.vocab
        token_max = word2vec_comment.syn0.shape[0]

        trees = pd.read_json(data_path)
        trees['code'] = trees['code'].apply(trans2seq)
        trees['comment'] = trees['comment'].apply(comment_to_index)
        trees.to_pickle(os.path.join(self.root, part, "block.pkl"))

    def run(self):
        self.dictionary_and_embedding(self.path)
        self.generate_block_seqs(self.path)


if __name__ == "__main__":
    root = "/home/cheng/Documents/projects/deep_comment_generation/data"
    train_path = "/home/cheng/Documents/projects/deep_comment_generation/data/train.json"
    valid_path = "/home/cheng/Documents/projects/deep_comment_generation/data/valid.json"
    test_path = "/home/cheng/Documents/projects/deep_comment_generation/data/test.json"
    part = "train" 
    identifier_dim_size = 128
    token_dim_size = 128
    max_identifier_voc_size = 100000
    max_token_voc_size = 100000
    ppl = Pipeline(root, part, identifier_dim_size, token_dim_size,path=train_path,max_identifier_voc_size=max_identifier_voc_size, max_token_voc_size=max_token_voc_size)
    ppl.run() 