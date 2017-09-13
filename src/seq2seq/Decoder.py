import torch

class Decoder(torch.nn.Module):
	def __init__(self, hidden_size, output_size, n_layers, embedding_matrix=None, embedding_padding_idx=None):
		super(Decoder, self).__init__()
		#self.hidden_size = hidden_size
		#self.n_layers = n_layers

		self.embedding = torch.nn.Embedding(output_size, hidden_size, padding_idx=embedding_padding_idx)
		if embedding_matrix is not None:
			self.embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix).float())
	
		self.gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
		self.out = torch.nn.Linear(hidden_size, output_size)
		self.softmax = torch.nn.LogSoftmax()

	# Input shape: (batch_size, 1)
	# Hidden shape: (batch_size, 1, hidden_size)
	# Output shape: (batch_size, output_size)
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
