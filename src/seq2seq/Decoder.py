import torch

class Decoder(torch.nn.Module):
	def __init__(self, hidden_size, output_size, n_layers):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		self.embedding = torch.nn.Embedding(output_size, hidden_size)
		self.lstm = torch.nn.LSTM(hidden_size, hidden_size)
		self.out = torch.nn.Linear(hidden_size, output_size)
		self.softmax = torch.nn.LogSoftmax()

	def forward(self, input, hidden):
		output = self.embedding(input).view(1, 1, -1)
		for i in range(self.n_layers):
			output = torch.nn.functional.relu(output)
			output, hidden = self.lstm(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def init_hidden(self):
		return torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size))
