import pandas as pd
import os
from nltk import word_tokenize
import xml.etree.ElementTree as ET


class Pipeline:
    def __init__(self, root):
        self.root = root
        self.sources = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.identifier_dim_size = None
        self.token_dim_size = None
        self.max_identifier_voc_size = 100000
        self.max_token_voc_size = 100000

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, output_file, size,):
        self.size = size
        if not input_file:
            input_file = self.train_file_path
        trees = pd.read_json(input_file)

        from prepare_data import get_ast_sequences

        def trans_to_sequence(ast):
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
        trees.to_json(os.path.join(self.root, "programs_ns.json"))

        from gensim.models.word2vec import Word2Vec
        # code
        w2v_code = Word2Vec(code_corpus, size=self.identifier_dim_size,
                            workers=8, sg=1, min_count=3, window=5, max_final_vocab=self.max_identifier_voc_size)
        w2v_code.save(os.path.join(
            self.root, 'identifier_w2v_'+str(self.identifier_dim_size)))

        # comment
        w2v_comment = Word2Vec(comment_corpus, size=self.token_dim_size,
                               workers=8, sg=1, min_count=3, window=5, max_final_vocab=self.max_token_voc_size)
        w2v_comment.save(os.path.join(
            self.root, 'token_w2v_'+str(self.token_dim_size)))

    # generate block sequences with index representations
    def generate_block_seqs(self, data_path, part):
        from prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        # code word2vec vocabulary
        word2vec_code = Word2Vec.load(
            self.root, 'identifier_w2v_'+str(self.identifier_dim_size)).wv
        vocab_code = word2vec_code.vocab
        identifier_max = word2vec_code.syn0.shape[0]

        # comment word2vec vocabulary
        word2vec_comment = Word2Vec.load(
            self.root, 'token_w2v_'+str(self.token_dim_size)).wv
        vocab_comment = word2vec_comment.vocab
        token_max = word2vec_comment.syn0.shape[0]

        def comment_to_index(comment):
            return [vocab_comment[token] if token in vocab_comment else token_max for token in comment]

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

        trees = pd.read_pickle(data_path)
        trees['code'] = trees['code'].apply(trans2seq)
        trees['comment'] = trees['comment'].apply(comment_to_index)
        trees.to_pickle(os.path.join(self.root, part+"_block.pkl"))
