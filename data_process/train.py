from model import *
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
import time
import math
import os


# init the model parameters
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal(param.data, mean=0, std=0.01)
        else:
            nn.init.constant(param.data, 0)


def get_batch(dataset, idx, bs, comment_pad_index):
    tmp = dataset.iloc[idx: idx+bs]
    code, comment = [], []
    for _, item in tmp.iterrows():
        length =len(item)
        code.append(item[0])
        comment.append(item[1])
    # code doesn't need pad
    # pad comment
    trg_lens = [len(com) for com in comment]
    max_len = max(trg_lens)
    pad_trg_seq = [[comment_pad_index] *
                   (max_len - com_len) for com_len in trg_lens]
    for i in range(len(comment)):
        comment[i].extend(pad_trg_seq[i])

    comment = np.asarray(comment)

    return code, comment


def train(model, dataset, trg_pad_index, optimizer, criterion, clip):
    print("start training")
    model.train()
    epoch_loss = 0

    start_index=10000
    i = start_index
    while i < len(dataset):
        if i < 100:
            batch_size=1
        elif i < 2000:
            batch_size=2
        elif i < 10000:
            batch_size=5
        elif i < 20000:
            batch_size=10
        elif i < 30000:
            batch_size=20
        else:
            batch_size =60

        batch = get_batch(dataset, i, batch_size, trg_pad_index)
        code, comment = batch
        i += batch_size

        # code = [batch, None]
        # comment =[batch, max_len]

        optimizer.zero_grad()
        output = model(code, comment)  # output = [max_len, batch, output_dim]
        output_dim = output.shape[-1]
        # negelect the first token <sos>
        output = output[1:, :, :].view(-1, output_dim)
        comment = comment[:, 1:].reshape(-1)
        comment=torch.from_numpy(comment).cuda()

        # output = [(max_len-1)* batch, output_dim]
        # comment = [(max_len-1)* batch]

        loss = criterion(output, comment)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        
        print("training step: ",i - start_index )

    return epoch_loss / len(dataset-start_index)


def evaluate(model, dataset, trg_pad_index, criterion):
    print("start evaluate")
    model.eval()
    epoch_loss = 0
    
    start_index=10
    i = start_index
    with torch.no_grad():
        while i < len(dataset):
            if i < 500:
                batch_size=1
            elif i < 1000:
                batch_size=2
            elif i <2000:
                batch_size=10
            elif i < 30000:
                batch_size=20
            else:
                batch_size=40

            batch = get_batch(dataset, i,batch_size, trg_pad_index)
            code, comment = batch
            i += batch_size

            output = model(code, comment, 0)  # turn off the teacher forcing

            output_dim = output.shape[-1]

            output = output[:, 1:, :].view(-1, output_dim)
            comment = comment[:, 1:].reshape(-1)
            comment=torch.from_numpy(comment).cuda()

            loss = criterion(output, comment)

            epoch_loss += loss.item()

            print("evaluate step: ", i )


    return epoch_loss / len(dataset)


def epoch_time(start, end):
    elapsed_time = end-start
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":

    root = '/home/cheng/Documents/projects/deep_comment_generation/data'
    train_data = pd.read_pickle(os.path.join(root, "test", "block.pkl"))
    val_data = pd.read_pickle(os.path.join(root, "valid", "block.pkl"))
    test_data = pd.read_pickle(os.path.join(root, "test", "block.pkl"))

    identifier_word2vec = Word2Vec.load(os.path.join(root, "train", "identifier_w2v_128")).wv
    token_word2vec = Word2Vec.load(os.path.join(root, "train", "token_w2v_128")).wv

    identifier_embeddings = np.zeros((identifier_word2vec.syn0.shape[0] + 1, identifier_word2vec.syn0.shape[1]), dtype="float32")
    identifier_embeddings[:identifier_word2vec.syn0.shape[0]] = identifier_word2vec.syn0

    token_embeddings = np.zeros((token_word2vec.syn0.shape[0] + 1, token_word2vec.syn0.shape[1]), dtype="float32")
    token_embeddings[:token_word2vec.syn0.shape[0]] = token_word2vec.syn0

    USE_GPU=True
    IDENTIFIER_DIM = identifier_word2vec.syn0.shape[1]
    ENC_RNN_HID_DIM = 128
    IDENTIFIER_VOC_SIZE = len(identifier_embeddings)
    ENCODE_DIM = 128
    BATCH_SIZE = 5

    DEC_RNN_HID_DIM = 128
    TRG_VOC_SIZE = len(token_embeddings)
    TRG_EMBEDDING_SIZE = token_word2vec.syn0.shape[1]
    TRG_PAD_IDX = token_word2vec.vocab["<pad>"].index
    ENC_DROPOUT = 0.7
    OUTPUT_DROPOUT = 0.7
    N_EPOCH = 10
    CLIP = 1

    best_valid_loss = float('inf')

    encoder = Encoder(IDENTIFIER_DIM, ENC_RNN_HID_DIM, IDENTIFIER_VOC_SIZE, ENCODE_DIM, DEC_RNN_HID_DIM, True, identifier_embeddings)
    attn = Attention(ENC_RNN_HID_DIM, DEC_RNN_HID_DIM)
    decoder = Decoder(TRG_EMBEDDING_SIZE,ENC_RNN_HID_DIM, DEC_RNN_HID_DIM, TRG_VOC_SIZE, OUTPUT_DROPOUT, token_embeddings, attn)
    model = Seq2Seq(encoder, decoder, USE_GPU)
    init_weights(model)
    
    if USE_GPU:
        model.cuda()

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    for epoch in range(N_EPOCH):
        start_time = time.time()

        train_loss = train(model, val_data, TRG_PAD_IDX, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_data, TRG_PAD_IDX, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "comment-generation-model.pt")                

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # load parameters from best validation loss and get result on the test dataset

    model.load_state_dict(torch.load('comment-generation-model.pt'))
    test_loss = evaluate(model, test_data, TRG_PAD_IDX, criterion)

    print(f'| Test Loss:{test_loss:.3f} | Test PPL:{math.exp(test_loss):7.3f} |')
