import torch
import torch.nn.functional as F

from drift import GUMBEL_DIST
from drift.core import BaseSpeaker, BaseListener


class Speaker(BaseSpeaker):
    def __init__(self, env_config):
        super(Speaker, self).__init__(env_config)
        self.hidden_size = 50
        self.len = self.env_config['p']
        self.obj_to_h = torch.nn.Linear(self.env_config['p'] * self.env_config['t'], self.hidden_size)
        self.embeddings = torch.nn.Embedding(num_embeddings=self.env_config['p'] * self.env_config['t'],
                                             embedding_dim=self.hidden_size)
        #self.gru = torch.nn.GRUCell(self.hidden_size, self.hidden_size)
        # Not using GRUCell because the potential speed-up in teacher forcing
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        self.linear_dec = torch.nn.Linear(self.hidden_size, self.env_config['p'] * self.env_config['t'], bias=False)

        # This act as the embedding of <bos>
        self.init_input = torch.nn.Parameter(torch.rand(1, self.hidden_size), requires_grad=True)
        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.linear_dec.weight, std=0.1)
        torch.nn.init.normal_(self.obj_to_h.weight, std=0.1)
        torch.nn.init.zeros_(self.obj_to_h.bias)
        torch.nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)

    def get_logits(self, objs, msgs):
        """ Teacher forcing
        :return logits: [bsz, len, hidden_size]
        """
        # init emb [bsz, hidden_size]
        # inp_embs [bsz, len, msgs]. Remove the last time step
        bsz = objs.shape[0]
        init_inputs = self._prepare_init_input(bsz)
        msgs_embs = self.embeddings(msgs)
        inp_embs = torch.cat([init_inputs[:, None, :], msgs_embs[:, :-1]], dim=1)

        # Enc obj to state [bsz, hidden_size]
        oh_objs = self._one_hot(objs)
        states = self.obj_to_h(oh_objs)
        rnn_out, _ = self.gru(inp_embs, states.unsqueeze(0))
        logits = self.linear_dec(rnn_out)
        return logits

    def greedy(self, objs):
        # inputs [bsz, hidden_size]
        bsz = objs.shape[0]
        inputs = self._prepare_init_input(bsz)

        # Enc obj to state [bsz, hidden_size]
        oh_objs = self._one_hot(objs)
        states = self.obj_to_h(oh_objs)
        msgs = []
        for t in range(self.len):
            out, states = self._step_gru(inputs, states)
            logits = self.linear_dec(out)
            words = torch.argmax(logits, dim=-1)
            msgs.append(words)

            # Prepare next inputs
            inputs = self.embeddings(words)
        msgs = torch.stack(msgs, dim=1)
        return msgs

    def sample(self, objs):
        """
        :return: logprobs [bsz, len]
                 msgs [bsz, len]
        """
        # inputs [bsz, hidden_size]
        bsz = objs.shape[0]
        inputs = self._prepare_init_input(bsz)

        # Enc obj to state [bsz, hidden_size]
        oh_objs = self._one_hot(objs)
        states = self.obj_to_h(oh_objs)
        msgs = []
        logprobs = []
        for t in range(self.len):
            out, states = self._step_gru(inputs, states)
            logits = self.linear_dec(out)
            dist = torch.distributions.Categorical(logits=logits)
            words = dist.sample()
            msgs.append(words)
            logprobs.append(dist.log_prob(words))

            # Prepare next inputs
            inputs = self.embeddings(words)
        msgs = torch.stack(msgs, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        return logprobs, msgs

    def gumbel(self, objs, temperature=1):
        """
        :return: y [bsz, len, vocab_size]
                 msgs [bsz, len]
        """
        # inputs [bsz, hidden_size]
        bsz = objs.shape[0]
        inputs = self._prepare_init_input(bsz)

        # Enc obj to state [bsz, hidden_size]
        oh_objs = self._one_hot(objs)
        states = self.obj_to_h(oh_objs)
        msgs = []
        ys = []
        for t in range(self.len):
            out, states = self._step_gru(inputs, states)
            logits = self.linear_dec(out)
            logprobs = F.log_softmax(logits, dim=-1)
            g = GUMBEL_DIST.sample(logits.shape)
            y = F.softmax((g + logprobs) / temperature, dim=-1)
            words = torch.argmax(y, dim=-1)
            msgs.append(words)
            ys.append(y)

            # Prepare next inputs
            inputs = self.embeddings(words)
        msgs = torch.stack(msgs, dim=1)
        ys = torch.stack(ys, dim=1)
        return ys, msgs

    def _one_hot(self, objs):
        """ Make input a concatenation of one-hot
        :param objs [bsz, nb_props]
        :param oh_objs [bsz, nb_props * nb_types]
        """
        oh_objs = torch.Tensor(size=[objs.shape[0], objs.shape[1], self.env_config['t']])
        oh_objs = oh_objs.to(device=objs.device)
        oh_objs.zero_()
        oh_objs.scatter_(2, objs.unsqueeze(-1), 1)
        return oh_objs.view([objs.shape[0], -1])

    def _prepare_init_input(self, batch_size):
        """ Repeating the first dim """
        return self.init_input.repeat([batch_size, 1])

    def _step_gru(self, inputs, states):
        """ Step our gru
        :param inputs: [bsz, hidden_size]
        :param states: [bsz, hidden_size]
        :return out: [bsz, hidden_size],
                next_states: [bsz, hidden_size]
        """
        out, next_states = self.gru(inputs.unsqueeze(1), states.unsqueeze(0))
        return out.squeeze(1), next_states.squeeze(0)


class Listener(BaseListener):
    def __init__(self, env_config):
        super(Listener, self).__init__(env_config)
        self.hidden_size = 50
        self.len = self.env_config['p']

        # Effectively an embedding
        self.linear_in = torch.nn.Linear(self.env_config['p'] * self.env_config['t'],
                                         self.hidden_size, bias=False)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # A learnable parameter for init states
        self.init_states = torch.nn.Parameter(torch.rand(1, 1, self.hidden_size), requires_grad=True)
        self.linear_out = torch.nn.Linear(self.hidden_size, self.env_config['t'])
        self.init_weight()

    def init_weight(self):
        torch.nn.init.uniform_(self.linear_in.weight, -0.1, 0.1)
        torch.nn.init.normal_(self.linear_out.weight, std=0.1)
        torch.nn.init.zeros_(self.linear_out.bias)

    def get_logits(self, oh_msgs):
        """ 
        oh_msgs: [bsz, nb_prop, vocab_size]
        return: [bsz, nb_prop, type_size]
        """
        bsz = oh_msgs.shape[0]
        init_states = self._prepare_init_states(bsz)
        inp_embs = self.linear_in(oh_msgs)
        rnn_out, _ = self.gru(inp_embs, init_states)
        return self.linear_out(rnn_out)

    def _prepare_init_states(self, batch_size):
        return self.init_states.repeat(1, batch_size, 1)
