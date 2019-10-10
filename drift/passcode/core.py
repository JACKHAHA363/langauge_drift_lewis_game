import torch
import attr


class Model(torch.nn.Module):
    """ Just a 2-layer MLP """
    def __init__(self, nb_code, model_size):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(nb_code, model_size)
        self.linear = torch.nn.Linear(model_size, model_size)
        self.dec_linear = torch.nn.Linear(model_size, nb_code)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        """
        :param inputs: [bsz]
        :return: [bsz, nb_code]. logits
        """
        return self.dec_linear(self.relu(self.linear(self.emb(inputs))))

    def get_action_dist(self, inputs):
        """ return a distribution object """
        logits = self.forward(inputs)
        return torch.distributions.Categorical(logits=logits)

    def save(self, pth_path):
        info = {'nb_code': self.emb.num_embeddings,
                'model_size': self.emb.embedding_dim,
                'state_dict': self.state_dict()}
        torch.save(info, pth_path)

    @classmethod
    def load(cls, pth_path):
        """ Load from pth path
            Usage: model = Model.load('model.pth')
        """
        info = torch.load(pth_path)
        model = cls(nb_code=info['nb_code'], model_size=info['model_size'])
        model.load_state_dict(info['state_dict'])
        return model


@attr.s
class Batch:
    # The codes, messages and decoded results
    codes = attr.ib()
    msg = attr.ib()
    decs = attr.ib()

    # Distribution of sender and receiver
    s_dist = attr.ib()
    r_dist = attr.ib()


class Env:
    """ code_size == msg_size """
    def __init__(self, nb_code):
        self.nb_code = nb_code

    def generate_trajs(self, sender, receiver, batch_size):
        """ Given two model generate a batch of trajs
        """
        codes = torch.randint(low=0, high=self.nb_code, size=[batch_size])
        s_dist = sender.get_action_dist(codes)
        msg = s_dist.sample()

        r_dist = receiver.get_action_dist(msg)
        decs = r_dist.sample()
        return Batch(codes=codes, msg=msg, decs=decs, s_dist=s_dist, r_dist=r_dist)


def eval_drift(model, nb_codes):
    """ Check how many symbols mismatched """
    codes = torch.range(0, nb_codes-1).long()
    with torch.no_grad():
        logits = model(codes)
    decs = torch.argmax(logits, -1)
    return (codes != decs).float().mean().item()


def eval_comm(sender, receiver, nb_codes):
    codes = torch.range(0, nb_codes - 1).long()
    with torch.no_grad():
        sender_logits = sender(codes)
        msg = torch.argmax(sender_logits, -1)
        receiver_logits = receiver(msg)
        decs = torch.argmax(receiver_logits, -1)
    succ = (decs == codes)
    succ_rate = succ.float().mean()
    incorrect_msg = (codes != msg)
    fp = succ & incorrect_msg
    fp_rate = fp.float().mean()
    return succ_rate, fp_rate
