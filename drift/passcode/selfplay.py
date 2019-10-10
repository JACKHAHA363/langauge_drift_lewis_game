from drift.passcode.core import Model, eval_drift, eval_comm
import torch
from tensorboardX import SummaryWriter

PRETRAIN_CKPT_STEPS = 100
PRETRAIN_CKPT = 'pretrain_step{}.pth'.format(PRETRAIN_CKPT_STEPS)
BATCH_SIZE = 20
SLR = 1e-3
RLR = 1e-3
STEPS = 6000
writer = SummaryWriter('log_sp_slr{}_rlr{}_pretrain{}'.format(SLR, RLR,
                                                              PRETRAIN_CKPT_STEPS))

sender = Model.load(PRETRAIN_CKPT)
sopt = torch.optim.Adam(lr=SLR, params=sender.parameters())

receiver = Model.load(PRETRAIN_CKPT)
ropt = torch.optim.Adam(lr=RLR, params=receiver.parameters())

NB_CODE = sender.emb.num_embeddings

for step in range(STEPS):
    codes = torch.randint(low=0, high=NB_CODE, size=[BATCH_SIZE]).long()
    s_dist = sender.get_action_dist(codes)
    msg = s_dist.sample()
    r_dist = receiver.get_action_dist(msg)
    decs = r_dist.sample()
    rewards = (codes == decs).float()

    # Update sender
    s_logprobs = s_dist.log_prob(msg)
    s_reinforce = (s_logprobs * rewards).mean()
    sopt.zero_grad()
    (-s_reinforce).backward()
    sopt.step()

    # Update receiver
    r_logprobs = r_dist.log_prob(decs)
    r_reinforce = (r_logprobs * rewards).mean()
    ropt.zero_grad()
    (-r_reinforce).backward()
    ropt.step()

    s_drift_score = eval_drift(sender, NB_CODE)
    r_drift_score = eval_drift(receiver, NB_CODE)
    succ_rate, fp_rate = eval_comm(sender, receiver, NB_CODE)

    stats = {'drift_score/sender': s_drift_score,
             'drift_score/receiver': r_drift_score,
             'succ_rate': succ_rate, 'fp_rate': fp_rate,
             'rwd': rewards.mean().item()}
    if step % 10 == 0:
        logstr = []
        for name, val in stats.items():
            writer.add_scalar(name, val, step)
            logstr.append('{}:{:.4f}'.format(name, val))
        print(' '.join(logstr))
