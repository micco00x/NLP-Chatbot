import torch

class Encoder(torch.nn.Module):
	def __init__(self, input_size, hidden_size, n_layers, embedding_matrix=None):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		self.embedding = torch.nn.Embedding(input_size, hidden_size)
		if embedding_matrix != None:
			self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

		self.lstm = torch.nn.LSTM(hidden_size, hidden_size)

	def forward(self, input, hidden):
		output = self.embedding(input).view(1, 1, -1)
		for i in range(self.n_layers):
			output, hidden = self.lstm(output, hidden)
		return output, hidden

	def initHidden(self):
		return torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size))
