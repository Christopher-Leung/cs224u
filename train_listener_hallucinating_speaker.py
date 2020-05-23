from colors import ColorsCorpusReader
import os
from sklearn.model_selection import train_test_split
from torch_color_selector import (
    ColorizedNeuralListener, create_example_dataset)
from torch_listener_with_attention import (
    AttentionalColorizedNeuralListener, create_example_dataset)
from torch_color_describer import ColorizedInputDescriber
import utils
from utils import START_SYMBOL, END_SYMBOL, UNK_SYMBOL
import numpy as np
import torch
import sys

utils.fix_random_seeds()

def load_from_pickle():
    import pickle 
    
    with open('dev_vocab.pickle', 'rb') as handle:
        dev_vocab = pickle.load(handle)
    with open('dev_vocab_speaker.pickle', 'rb') as handle:
        dev_vocab_speaker = pickle.load(handle)
    with open('dev_vocab_listener.pickle', 'rb') as handle:
        dev_vocab_listener = pickle.load(handle)
    with open('dev_seqs_test.pickle', 'rb') as handle:
        dev_seqs_test = pickle.load(handle)
    with open('dev_seqs_train.pickle', 'rb') as handle:
        dev_seqs_train = pickle.load(handle)
    with open('dev_seqs_train_listener.pickle', 'rb') as handle:
        dev_seqs_train_listener = pickle.load(handle)
    with open('dev_seqs_train_speaker.pickle', 'rb') as handle:
        dev_seqs_train_speaker = pickle.load(handle)
    with open('dev_cols_test.pickle', 'rb') as handle:
        dev_cols_test = pickle.load(handle)
    with open('dev_cols_train.pickle', 'rb') as handle:
        dev_cols_train = pickle.load(handle)
    with open('dev_cols_train_listener.pickle', 'rb') as handle:
        dev_cols_train_listener = pickle.load(handle)
    with open('dev_cols_train_speaker.pickle', 'rb') as handle:
        dev_cols_train_speaker = pickle.load(handle)
    with open('dev_examples_test.pickle', 'rb') as handle:
        dev_examples_test = pickle.load(handle)
    with open('embedding.pickle', 'rb') as handle:
        embedding = pickle.load(handle)
    return dev_vocab, dev_vocab_speaker, dev_vocab_listener, dev_seqs_test, dev_seqs_train, dev_seqs_train_speaker, dev_seqs_train_listener, dev_cols_test, dev_cols_train, dev_cols_train_speaker, dev_cols_train_listener, dev_examples_test, embedding

def load_glove_from_pickle():
    import pickle 
    with open('dev_glove_vocab.pickle', 'rb') as handle:
        dev_glove_vocab = pickle.load(handle)
    with open('dev_glove_embedding.pickle', 'rb') as handle:
        dev_glove_embedding = pickle.load(handle)
    return dev_glove_vocab, dev_glove_embedding

## Load dataset

dev_vocab, dev_vocab_speaker, dev_vocab_listener, dev_seqs_test, dev_seqs_train, dev_seqs_train_speaker,     dev_seqs_train_listener, dev_cols_test, dev_cols_train, dev_cols_train_speaker, dev_cols_train_listener,     dev_examples_test, embedding = load_from_pickle()
dev_glove_vocab, dev_glove_embedding = load_glove_from_pickle()

# Load listener models
literal_listener_listener = ColorizedNeuralListener(
dev_vocab_listener, 
#embedding=dev_glove_embedding, 
embed_dim=100,
embedding=embedding,
hidden_dim=100, 
max_iter=100,
batch_size=256,
dropout_prob=0.,
eta=0.001,
lr_rate=0.96,
warm_start=True,
device='cuda')
literal_listener_listener.load_model("literal_listener_with_attention_listener_split.pt")

literal_listener_speaker = ColorizedNeuralListener(
dev_vocab_speaker, 
#embedding=dev_glove_embedding, 
embed_dim=100,
embedding=embedding,
hidden_dim=100, 
max_iter=100,
batch_size=256,
dropout_prob=0.,
eta=0.001,
lr_rate=0.96,
warm_start=True,
device='cuda')
literal_listener_speaker.load_model("literal_listener_with_attention_speaker_split.pt")

def train_and_save(alpha=0, speaker_preference=0.5):
    print("Training with alpha:",alpha,"speaker_preference:",speaker_preference)
    alpha = float(alpha)
    speaker_preference = float(speaker_preference)
        
    literal_speaker = ColorizedInputDescriber(
    dev_glove_vocab, 
    embedding=dev_glove_embedding, 
    hidden_dim=100, 
    max_iter=40, 
    eta=0.0005,
    batch_size=32)
    literal_speaker.load_model("literal_speaker.pt")

    listener_hallucinating_speaker = ColorizedInputDescriber(
        dev_glove_vocab, 
        embedding=dev_glove_embedding, 
        hidden_dim=100, 
        max_iter=40, 
        eta=0.0005,
        batch_size=32,
        warm_start=True)
    listener_hallucinating_speaker.load_model("literal_speaker.pt")
    listener_hallucinating_speaker.warm_start=True
    listener_hallucinating_speaker.opt = listener_hallucinating_speaker.optimizer(
                    listener_hallucinating_speaker.model.parameters(),
                    lr=listener_hallucinating_speaker.eta,
                    weight_decay=listener_hallucinating_speaker.l2_strength)
    listener_hallucinating_speaker.max_iter=5
    
    
    num_hallucinations = 1
    m_samples = 3
    dataset = dev_cols_train_speaker
    utterances = listener_hallucinating_speaker.generate_listener_augmentations(dataset,\
                                                                                 literal_listener_speaker,
                                                                                 num_hallucinations=num_hallucinations,
                                                                                 k_samples=6, 
                                                                                 m_samples=m_samples, 
                                                                                 batch_size=1000, 
                                                                                 max_length=12,
                                                                                 alpha=1.,
                                                                                 speaker_preference=0.5)
    # Flatten
    top_hallucinations = [seq for seqs in utterances for seq in seqs]
    #top_hallucinations = utterances
    dev_cols_train_speaker_extended = [cols for cols in dataset for i in range(num_hallucinations)]
    
    # Fit
    for i in range(9):
        listener_hallucinating_speaker.fit(dev_cols_train_speaker_extended, top_hallucinations)
        listener_hallucinating_speaker.calc_performance(literal_listener_listener, dev_cols_test)
        
    listener_hallucinating_speaker.save_model("trained/listener_hallucinating_speaker_alpha"+str(alpha)+"_beta"+str(speaker_preference)+".pt")
    
if __name__ == '__main__':
    for alpha in [0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.]:
        for beta in [0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.]:
            train_and_save(alpha, beta)
            torch.cuda.empty_cache()
