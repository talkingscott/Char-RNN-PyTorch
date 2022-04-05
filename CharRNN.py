"""
Minimalist code for character-level language modelling using Multi-layer
Recurrent Neural Networks (LSTM) in PyTorch. The RNN is trained to predict next
letter in a given text sequence. The trained model can then be used to generate
a new text sequence resembling the original data.

Forked from https://github.com/nikhilbarhate99/Char-RNN-PyTorch
"""

from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

DIVIDER = '-' * 72
STOP_UNSAVED_COUNT = 3

device_type = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_type}")
device = torch.device(device_type)


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers,
                 dropout):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())


def train():
    # +Hyperparameters
    hidden_size = 512   # size of hidden state
    seq_len = 100       # length of LSTM sequence
    num_layers = 3      # num of layers in LSTM layer stack
    dropout = 0.1       # dropout probability
    lr = 0.002          # learning rate
    epochs = 100        # max number of epochs
    op_seq_len = 300    # total num of characters in output test sequence
    load_chk = False    # load weights from save_path to continue training
    save_path = "./preTrained/CharRNN_shakespeare.pth"
    data_path = "./data/shakespeare.txt"
    # -Hyperparameters

    # load the text file
    data_str = open(data_path, 'r').read()
    chars = sorted(list(set(data_str)))
    data_size, vocab_size = len(data_str), len(chars)
    print(DIVIDER)
    print("Data has {} characters, {} unique".format(data_size, vocab_size))
    print(DIVIDER)

    # char to index and index to char maps
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    # convert data from chars to indices
    data_list = [-1 for i in range(data_size)]
    for i, ch in enumerate(data_str):
        data_list[i] = char_to_ix[ch]

    # data tensor on device
    data = torch.tensor(data_list).to(device)
    data = torch.unsqueeze(data, dim=1)

    # model instance
    rnn = (RNN(vocab_size, vocab_size, hidden_size, num_layers, dropout)
           .to(device))

    # load checkpoint if True
    if load_chk:
        rnn.load_state_dict(torch.load(save_path))
        print("Model loaded successfully !!")
        print(DIVIDER)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

    min_loss = float("inf")
    unsaved_count = 0

    # training loop
    for i_epoch in range(1, epochs+1):

        # random starting point (1st seq_len chars) from data to begin
        data_ptr = np.random.randint(seq_len)
        steps = 0
        running_loss = 0
        hidden_state = None
        start_time = time()

        while True:
            input_seq = data[data_ptr:data_ptr+seq_len]
            target_seq = data[data_ptr+1:data_ptr+seq_len+1]

            # forward pass
            output, hidden_state = rnn(input_seq, hidden_state)

            # compute loss
            loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
            running_loss += loss.item()

            # compute gradients and take optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the data pointer
            data_ptr += seq_len
            steps += 1

            # if at end of data : break
            if data_ptr + seq_len + 1 > data_size:
                break

        # print loss and save weights after every epoch
        # TODO: don't rely on training loss to decide when to stop,
        # use validation loss instead.
        print("Epoch: {0} \t Loss: {1:.8f} \t Duration: {2:.3f} seconds"
              .format(i_epoch, running_loss/steps, time() - start_time))
        if running_loss < min_loss:
            print("Save weights")
            torch.save(rnn.state_dict(), save_path)
            min_loss = running_loss
            unsaved_count = 0
        else:
            print("Do not save weights")
            unsaved_count += 1
            if unsaved_count >= STOP_UNSAVED_COUNT:
                break

        # sample / generate a text sequence after every epoch
        sample_len = 0
        hidden_state = None

        # random character from data to begin
        rand_index = np.random.randint(data_size-1)
        input_seq = data[rand_index:rand_index+1]

        print(DIVIDER)
        while sample_len < op_seq_len:
            # forward pass
            output, hidden_state = rnn(input_seq, hidden_state)

            # construct categorical distribution and sample a character
            output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(output)
            index = dist.sample()

            # print the sampled character
            print(ix_to_char[int(index.item())], end='')

            # next input is current output
            input_seq[0][0] = index.item()
            sample_len += 1

        print("\n")
        print(DIVIDER)


if __name__ == '__main__':
    train()
