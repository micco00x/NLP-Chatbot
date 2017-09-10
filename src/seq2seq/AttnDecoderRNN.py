import torch

class AttnDecoderRNN(torch.nn.Module):
	def __init__(self, hidden_size, output_size, max_length, n_layers=1, dropout_p=0.1, embedding_matrix=None):
		super(AttnDecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout_p = dropout_p
		self.max_length = max_length

		self.embedding = torch.nn.Embedding(self.output_size, self.hidden_size)
		if embedding_matrix is not None:
			self.embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix).float())
		
		self.attn = torch.nn.Linear(self.hidden_size * 2, self.max_length)
		self.attn_combine = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = torch.nn.Dropout(self.dropout_p)
		self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
		self.out = torch.nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.dropout(embedded)

		attn_weights = torch.nn.functional.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)))
		attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		for i in range(self.n_layers):
			output = torch.nn.functional.relu(output)
			output, hidden = self.gru(output, hidden)

		output = torch.nn.functional.log_softmax(self.out(output[0]))
		return output, hidden, attn_weights

	def initHidden(self):
		result = torch.autograd.Variable(torch.zeros(1, 1, self.hidden_size))
		if use_cuda:
			return result.cuda()
		else:
			return result
