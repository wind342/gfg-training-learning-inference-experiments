"""Fixed realized-displacement control; never rescale optimizer moments."""
import torch
from experiments.gfg_controller_resident_reuse_v1.engine import ResidentBuffers


class StaticControl:
    def __init__(self, model, alpha):
        self.alpha = alpha
        self.buffers = ResidentBuffers(model)
        self.identity_steps_skipped = 0

    def update(self, model, opt, t, x, y, vx, vy, seed, cached_alpha):
        model.train()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.alpha != 1.:
            self.buffers.capture('before')
        opt.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if self.alpha != 1.:
            self.buffers.form_delta()
            self.buffers.apply(self.alpha)
        return self.alpha, {'loss': float(loss.detach()), 'alpha': self.alpha}, None
