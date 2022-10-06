import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding_dim = embedding_size

        self.hidden = torch.zeros(1, 1, hidden_size)

        # self.embedding provides a vector representation of the inputs to our model
        self.embedding = nn.Embedding(num_embeddings=self.input_size,
                                      embedding_dim=self.embedding_dim)
        
        # self.lstm, accepts the vectorized input and passes a hidden state
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_size,
                            num_layers=1)
    # Forward pass
    def forward(self, i):
        """
        Inputs: i, the src vector
        Outputs: o, the encoder outputs
                h, the hidden state
                c, the cell state
        """
        embedded = self.embedding(i)
        
        # Since lstm returns o, h and c, we can directly return it
        return self.lstm(embedded)


# Decoder class
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        Self.embedding_size = embedding_size
        
        # self.embedding provides a vector representation of the target to our model
        self.embedding = nn.Embedding(num_embeddings=self.output_size,
                                      embedding_dim=self.embedding_size)
        
        # self.lstm, accepts the embeddings and outputs a hidden state
        self.lstm = nn.LSTM(self.embedding_size, hidden_size, num_layers=3)
        
        # self.output, predicts on the hidden state via a linear output layer
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
    # Forward pass
    def forward(self, i, h, c):
        '''
        Inputs: i, the target vector
        Outputs: o, the prediction
                h, the hidden state
        '''
        unsqueezed = i.unsqueeze(0)
        embedded = self.embedding(unsqueezed)
        
        o, h, c = self.lstm(embedded, (h, c))
        
        # p = self.fc(o.squeeze(0))
        
        return o, h, p, c
        

        
        
        
        
class Seq2Seq(nn.Module):
    
    def __init__(self, encoder_input_size, encoder_hidden_size, decoder_hidden_size, decoder_output_size):
        
        super(Seq2Seq, self).__init__()
        
    
    
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder_output_size
        
        # tensor to store decoder outputs
        o = torch.zeros(trg_len, batch_size, trg_vocab_size)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        h, c = self.encoder(src)
        
        # first input to the decoder is the <sos> token.
        input = trg[0, :]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states 
            # receive output tensor (predictions) and new hidden and cell states.
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # replace predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions.
            top1 = output.argmax(1)
            # update input : use ground_truth when teacher_force 
            input = trg[t] if teacher_force else top1
            
            
        return o

    

