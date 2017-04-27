import torch
from torch import nn, from_numpy
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x, self.initHidden(x.size()[0]))
        last = out[:, -1, :]
        return self.linear(last)

    def initHidden(self, N):
        return Variable(torch.randn(1, N, self.hidden_size))

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x, self.initHidden(x.size()[0]))
        last = out[:, -1, :]
        return self.linear(last)

    def initHidden(self, N):
        return Variable(torch.randn(1, N, self.hidden_size))

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):        
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x, self.initHidden(x.size()[0]))
        last = out[:, -1, :]
        return self.linear(last)

    def initHidden(self, N):
        return (Variable(torch.zeros(1, N, self.hidden_size)),
                Variable(torch.zeros(1, N, self.hidden_size)))
