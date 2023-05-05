import torch
from torch import nn

# NOTE: In addition to __init__() and forward(), feel free to add
# other functions or attributes you might need.

class DAN(torch.nn.Module):
    def __init__(self, glove, load_weights, hidden_dim=250, p_drop=0):
        # TODO: Declare DAN architecture
        super(DAN, self).__init__()

        if glove == 'glove':
            self.embedding = nn.Embedding.from_pretrained(load_weights, freeze=True)
        else:
            self.embedding = nn.Embedding.from_pretrained(load_weights, freeze=False)

        # first hidden layer
        self.hidden_layer = nn.Linear(2 * 50, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_drop)

        # second hidden layer
        # self.hidden_layer_2 = nn.Linear(hidden_dim,hidden_dim)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=p_drop)

        # output
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s1, s2):
        # TODO: Implement DAN forward pass
        embed1 = torch.mean(self.embedding(s1), dim=1)
        # print("embedding",self.embedding(s1).shape)
        # print("embed",embed1.shape)
        embed2 = torch.mean(self.embedding(s2), dim=1)
        cat_rep = torch.cat((embed1, embed2), 1)

        # first hidden layer
        hidden_rep = self.hidden_layer(cat_rep)
        relu_rep = self.relu(hidden_rep)
        # print("hidden",hidden_rep.shape)
        drop = self.dropout(relu_rep)
        # print("drop",drop.shape)

        # # second hidden layer
        # hidden_rep_2 = self.hidden_layer_2(cat_rep)
        # relu_rep_2 = self.relu(hidden_rep_2)
        # # print("hidden",hidden_rep.shape)
        # drop_2 = self.dropout(relu_rep_2)

        # output
        output = self.output_layer(drop)
        # print("ouput",output.shape)

        output = self.sigmoid(output)
        # print("sigmoid",output.shape)

        # print("return",output.squeeze(0).squeeze(0).shape)

        return output.squeeze(0).squeeze(0)


class RNN(torch.nn.Module):
    def __init__(self, rnn_bidirect, glove,load_weights, rnn_dim=10, layer_num=1, hidden_dim=20, p_drop=0.3):

        # TODO: Declare RNN model architecture
        super(RNN, self).__init__()


        if glove == 'glove':
            self.embedding = nn.Embedding.from_pretrained(load_weights, freeze=True)
        else:
            self.embedding = nn.Embedding.from_pretrained(load_weights, freeze=False)

        if rnn_bidirect:
            self.rnn = nn.RNN(50, rnn_dim, num_layers=layer_num, bias=False,
                              bidirectional=True)  # one layer, bidirectional
            self.hidden_layer = nn.Linear(4 * rnn_dim, hidden_dim)
        else:
            self.rnn = nn.RNN(50, rnn_dim, num_layers=layer_num, bias=False)  # two layers
            self.hidden_layer = nn.Linear(2 * rnn_dim, hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_drop)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s1, s2):
        # TODO: Implement RNN forward pass
        embed1 = self.embedding(s1).squeeze(0)
        rnn_out1, _ = self.rnn(embed1)

        embed2 = self.embedding(s2).squeeze(0)
        rnn_out2, _ = self.rnn(embed2)

        cat_rep = torch.cat((rnn_out1[-1, :].unsqueeze(0), rnn_out2[-1, :].unsqueeze(0)), 1)

        hidden_rep = self.hidden_layer(cat_rep)
        relu_rep = self.relu(hidden_rep)
        # print("hidden",hidden_rep.shape)
        drop = self.dropout(relu_rep)
        # print("drop",drop.shape)
        output = self.output_layer(drop)
        # print("ouput",output.shape)

        output = self.sigmoid(output)
        # print("sigmoid",output.shape)

        return output.squeeze(0).squeeze(0)
    
def argmax(vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()
    
# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

# for pos
class LSTM(torch.nn.Module):
    def __init__(self, rnn_bidirect, glove, load_weights, tag_to_ix, lstm_dim=10, layer_num=2, hidden_dim = 20,p_drop=0.1):

        # TODO: Declare LSTM model architecture
        super(LSTM, self).__init__()

        if glove == 'glove':
            self.embedding = nn.Embedding.from_pretrained(load_weights, freeze=True)
        else:
            self.embedding = nn.Embedding.from_pretrained(load_weights, freeze=False)

        if rnn_bidirect:
            self.lstm = nn.LSTM(50, lstm_dim, num_layers=layer_num, bias=False, bidirectional = True) # one layer, bidirectional
            self.hidden_layer = nn.Linear(4*lstm_dim,hidden_dim)
        else:
            self.lstm = nn.LSTM(50, lstm_dim, num_layers=layer_num, bias=False) # two layers
            self.hidden_layer = nn.Linear(2*lstm_dim,hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_drop)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _forward_alg(self, feats):
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # tensor([[-10000., -10000., -10000., -10000., -10000.]])

        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        # tensor([[-10000., -10000., -10000.,      0., -10000.]])

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        
        # alpha is a forward score
        # tensor(22.1316, grad_fn=<AddBackward0>)
        return alpha
    
    def _score_sentence(self, feats, tags):
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"

        # Gives the score of a provided tag sequence
        score = torch.zeros(1) # tensor([0.])
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        # tags tensor([3, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        # tensor([2.8724], grad_fn=<AddBackward0>)
        return score

    def _viterbi_decode(self, feats):
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"

        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
    
    def forward(self, s1, s2, pos1=torch.tensor([]), pos2=torch.tensor([])):
        # TODO: Implement LSTM forward pass
        #s1 torch.Size([1, 7])

        embed1 = self.embedding(s1).squeeze(0) # embed1 (sentence length, 50) tensor([[],[],[]])
        lstm_out1,(_,_) = self.lstm(embed1) # lstm_out1.shape torch.Size([sentence length, 20]) tensor([[],[],[]], grad_fn=<SqueezeBackward1>)
        
        #_get_lstm_features
        lstm_feats1 = self.hidden2tag(lstm_out1) #lstm_feats torch.Size([7, 47]) tensor([[],[],[]],grad_fn=<AddmmBackward0>)
        
        crfloss1=0
        tag_seq1=[]
        #crf train for s1 and pos1
        if (pos1.numel() != 0):
            forward_score1 = self._forward_alg(lstm_feats1)
            gold_score1 = self._score_sentence(lstm_feats1, pos1)
            crfloss1 = forward_score1-gold_score1
        #for non training, find the best path for s1 pos tags, given the features.
        else:
            score1, tag_seq1 = self._viterbi_decode(lstm_feats1)

        embed2 = self.embedding(s2).squeeze(0)
        lstm_out2,(_,_) = self.lstm(embed2)

        #_get_lstm_features
        lstm_feats2 = self.hidden2tag(lstm_out2)

        crfloss2=0
        tag_seq2=[]
        #crf train for s2 and pos2
        if (pos2.numel() != 0):
            forward_score2 = self._forward_alg(lstm_feats2)
            gold_score2 = self._score_sentence(lstm_feats2, pos2)
            crfloss2 = forward_score2-gold_score2
        #for non training, find the best path for s2 pos tags, given the features.
        else:
            score1, tag_seq2 = self._viterbi_decode(lstm_feats2)

        cat_rep = torch.cat((lstm_out1[-1,:].unsqueeze(0), lstm_out2[-1,:].unsqueeze(0)),1)
        hidden_rep = self.hidden_layer(cat_rep)
        relu_rep = self.relu(hidden_rep)
		# print("hidden",hidden_rep.shape)
        drop = self.dropout(relu_rep)
		# print("drop",drop.shape)
        output = self.output_layer(drop)
        # print("ouput",output.shape)

        output = self.sigmoid(output)
        # print("sigmoid",output.shape)

        return output.squeeze(0).squeeze(0), crfloss1, crfloss2, tag_seq1, tag_seq2
