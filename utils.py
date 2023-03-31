
from torch.utils.data import Dataset
import datasets
import torch
import torch.nn as nn
import random
MAX_SEQ_LENGTH=64
SOS_token = "<start>"



class LSTMDataset(Dataset):
    """
    This class is used to create a dataset for the LSTM model. 
    It takes a dataset from the HuggingFace library and creates a vocabulary from it. 
    It also adds padding and start and stop tokens to each sentence.

    Parameters
    ----------
    dataset : HuggingFace dataset
        The dataset to be used for training the model.
    max_seq_length : int
        The maximum length of a sentence. Sentences that are longer than this will be truncated and sentences that are shorter will be padded.
    
    Attributes
    ----------
    word_to_index : dict
        A dictionary that maps each word in the vocabulary to its index in the embedding matrix.
    index_to_word : dict
        A dictionary that maps each index in the embedding matrix to its respective word.
    pad_idx : int
        The index of the padding token in the embedding matrix.
    dataset : HuggingFace dataset
        The dataset to be used for training the model.
    max_seq_length : int
        The maximum length of a sentence. Sentences that are longer than this will be truncated and sentences that are shorter will be padded.
    
    """


    #final size 130, with start and stop token
    def __init__(self,
                 dataset,
                 max_seq_length: int
                ):
        
        self.max_seq_length = max_seq_length


        #add start and stop tokens to each sentence using map function
        dataset = dataset.map(lambda x: {"text": "<start> " + x["text"] + " <stop>"})
        #add padding to each sentence using map function
        dataset = dataset.map(lambda x: {"text": x["text"] + " <pad>" * (self.max_seq_length - len(x["text"].split()))})



        #create a vocabulary with the unique words in the dataset
        vocab = set()

        for sentence in dataset:
            sentence = sentence["text"]
            for word in sentence.split():
                vocab.add(word)
        
        #add special tokens to the vocabulary
        vocab.add("<pad>")
        vocab.add("<unk>")
        vocab.add("<start>")
        vocab.add("<stop>")

        #other possibility - using torchtext
        #vocab = datasets.text.build_vocab_from_iterator(dataset.map(lambda x: x["text"].split()).to_iterable(), min_freq=1)
        #vocab = datasets.text.Vocab(vocab, special_tokens=["<pad>", "<unk>", "<start>", "<stop>"])
        #self.word_to_index = vocab.get_stoi()
        #self.index_to_word = vocab.get_itos()



        #define a dictionary that simply maps tokens to their respective index in the embedding matrix
        self.word_to_index = {}
        self.index_to_word = {}
        for i, word in enumerate(vocab):
            self.word_to_index[word] = i
            self.index_to_word[i]= word
        
        
        self.pad_idx = self.word_to_index["<pad>"]
        self.dataset = dataset



        
    def tokenize(self, sentence):
        """
        This function takes a sentence and returns a tensor of the same length as the sentence, where each word is replaced by its index in the embedding matrix.
        If a word is not in the vocabulary, it is replaced by the index of the unknown token.

        Parameters
        ----------
        sentence : str
            The sentence to be tokenized.

        Returns
        -------
        sentence_vec : torch.Tensor
            A tensor of the same length as the sentence, where each word is replaced by its index in the embedding matrix.


        """
        
        T = len(sentence.split())
        sentence_vec = torch.zeros((T), dtype=torch.long)
        for i, word in enumerate(sentence.split()):
            if word not in self.word_to_index:
                sentence_vec[i] = self.word_to_index["<unk>"]
            else:
                sentence_vec[i] = self.word_to_index[word]
        return sentence_vec



    def __len__(self):
        
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
            Function that returns x and y for the model, where x is the input sentence and y is the target sentence. they have a difference of one word
            
            Parameters
            ----------
            idx : int
                The index of the sentence in the dataset.

            Returns
            -------
            x : torch.Tensor
                A tensor of the same length as the sentence -1, where each word is replaced by its index in the embedding matrix.
            y : torch.Tensor
                A tensor of the same length as the sentence -1, where each word is replaced by its index in the embedding matrix.

        """

        
        x = self.dataset[idx]["text"].split()[:-1]
        y = self.dataset[idx]["text"].split()[1:]
        x = self.tokenize(" ".join(x))
        y = self.tokenize(" ".join(y))
        return x, y





class VanillaLSTM(nn.Module):
    """
    This class is used to create a vanilla LSTM model.
    
    Parameters
    ----------
    vocab_size : int
        The size of the vocabulary.
    embedding_dim : int
        The dimension of the embedding matrix.
    hidden_dim : int
        The dimension of the hidden state.
    num_layers : int
        The number of layers in the LSTM.
    dropout_rate : float
        The dropout rate.
    model_name : str
        The name of the model.
    embedding_weights : torch.Tensor
        The embedding matrix.
    freeze_embeddings : bool
        Whether to freeze the embedding matrix or not.

    Attributes
    ----------
    num_layers : int
        The number of layers in the LSTM.
    hidden_dim : int
        The dimension of the hidden state.
    embedding_dim : int
        The dimension of the embedding matrix.
    name : str
        The name of the model.
    embedding_weights : torch.Tensor
        The embedding matrix.
    lstm : torch.nn.LSTM
        The LSTM layer.
    dropout : torch.nn.Dropout
        The dropout layer.
    fc : torch.nn.Linear
        The fully connected layer.

    """




    def __init__(self, vocab_size, 
                 embedding_dim,
                 hidden_dim,
                 num_layers, 
                 dropout_rate,
                 model_name,
                 embedding_weights=None,
                 freeze_embeddings=False):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.name = model_name

        # pass embedding weights if exist
        if embedding_weights is not None:
            
            self.embedding_weights = nn.Embedding.from_pretrained(embedding_weights, freeze=freeze_embeddings)
        else:  # train from scratch embeddings
            self.embedding_weights = nn.Embedding(vocab_size, embedding_dim)


        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id):
       
        output, (hidden, cell) = self.lstm(self.embedding_weights(input_id))
        output = self.dropout(output)
        output = self.fc(output)
        return output
    


