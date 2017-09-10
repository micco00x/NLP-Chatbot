import torch

class Encoder(torch.nn.Module):
	def __init__(self, input_size, hidden_size, n_layers, embedding_matrix=None):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		self.embedding = torch.nn.Embedding(input_size, hidden_size)
		if embedding_matrix is not None:
			self.embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix).float())

		self.gru = torch.nn.GRU(hidden_size, hidden_size)

	def forward(self, input, hidden):
		embedded = self.embedding(input).view(1, 1, -1)
		output = embedded
		for i in range(self.n_layers):
			output, hidden = self.gru(output, hidden)
		return output, hidden

	def initHidden(self):
		result = torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size))
		if torch.cuda.is_available():
			return result.cuda()
		else:
			return result
