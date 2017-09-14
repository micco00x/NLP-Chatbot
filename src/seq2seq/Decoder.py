import torch

class Decoder(torch.nn.Module):
	def __init__(self,
				 num_embeddings, embedding_dim,
				 hidden_size, n_layers,
				 embedding_matrix=None, embedding_padding_idx=None):
		super(Decoder, self).__init__()
		#self.n_layers = n_layers

		self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=embedding_padding_idx)
		if embedding_matrix is not None:
			self.embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix).float())
	
		self.gru = torch.nn.GRU(embedding_dim, hidden_size, batch_first=True)
		self.out = torch.nn.Linear(hidden_size, num_embeddings)
		self.softmax = torch.nn.LogSoftmax()

	# Input shape: (batch_size, 1)
	# Hidden shape: (batch_size, 1, hidden_size)
	# Output shape: (batch_size, num_embeddings)
	def forward(self, input, hidden):
		#output = self.embedding(input).view(1, 1, -1)
		output = self.embedding(input)
		#for i in range(self.n_layers):
		#	output = torch.nn.functional.relu(output)
		#	output, hidden = self.gru(output, hidden)
		#output = self.softmax(self.out(output[0]))
		#return output, hidden
		output = torch.nn.functional.relu(output)
		output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[:,0,:]))
		return output, hidden
