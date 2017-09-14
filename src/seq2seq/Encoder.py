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

	def forward(self, input):
		embedded = self.embedding(input)
		return self.gru(embedded)
