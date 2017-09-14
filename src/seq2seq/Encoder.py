import torch

class Encoder(torch.nn.Module):
	def __init__(self,
				 num_embeddings, embedding_dim,
				 hidden_size, n_layers,
				 embedding_matrix=None, embedding_padding_idx=None):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=embedding_padding_idx)
		if embedding_matrix is not None:
			self.embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix).float())

		self.gru = torch.nn.GRU(embedding_dim, hidden_size, n_layers, batch_first=True)

	def forward(self, input, hidden):
		#embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.embedding(input)
		#output = embedded
		#for i in range(self.n_layers):
		#	output, hidden = self.gru(output, hidden)
		#return output, hidden
		return self.gru(embedded, hidden)
	

	def initHidden(self, batch_size):
		result = torch.autograd.Variable(torch.zeros(batch_size, self.n_layers, self.hidden_size))
		if torch.cuda.is_available():
			return result.cuda()
		else:
			return result
