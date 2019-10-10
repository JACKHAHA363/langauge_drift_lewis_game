from drift.passcode.core import Model, eval_drift, eval_comm
import torch
from tensorboardX import SummaryWriter

NB_CODE = 100
MODEL_SIZE = 50
BATCH_SIZE = 20
STEPS = 300
CKPT_STEP = 10
CKPT_NAME = 'pretrain_step{}.pth'
writer = SummaryWriter('log_pretrain')

model = Model(nb_code=NB_CODE, model_size=MODEL_SIZE)
optim = torch.optim.Adam(lr=0.001, params=model.parameters())


for step in range(STEPS):
    codes = torch.randint(low=0, high=NB_CODE, size=[BATCH_SIZE]).long()
    dist = model.get_action_dist(codes)
    log_probs = dist.log_prob(codes)
    log_probs = log_probs.mean()

    optim.zero_grad()
    (-log_probs).backward()
    optim.step()

    # Evaluate
    drift_score = eval_drift(model, nb_codes=NB_CODE)
    succ_rate, fp_rate = eval_comm(model, model, nb_codes=NB_CODE)
    print('step {} drift score {:.4f} succ_rate {:.4f} fp_rate {:.4f}'.format(step, drift_score,
                                                                              succ_rate, fp_rate))

    if step % CKPT_STEP == 0:
        model.save(CKPT_NAME.format(step))
        writer.add_scalar('drift', drift_score, step)
        writer.add_scalar('succ_rate', succ_rate, step)
        writer.add_scalar('fp_rate', fp_rate, step)
