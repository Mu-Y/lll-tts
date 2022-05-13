import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import quadprog
from utils import to_gpu
import pdb

# Auxiliary functions useful for GEM's inner optimization.


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class GEM(nn.Module):
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 mem_data,
                 n_tasks,
                 hp):
        """
        mem_data: data loader of buffered prev samples
        n_tasks: num of total tasks. should include current task. e.g when training french
                 n_tasks should be 2.
        """
        super(GEM, self).__init__()
        self.hp = hp
        self.margin = hp.memory_strength
        self.n_tasks = n_tasks

        self.net = model
        self.criterion = criterion

        self.opt = optimizer

        # allocate episodic memory
        self.mem_data = mem_data

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks).cuda()

    def forward(self, x):
        src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = x

        # get teacher forcing ratio
        if self.hp.constant_teacher_forcing: tf = self.hp.teacher_forcing
        else: tf = cos_decay(max(global_step - self.hp.teacher_forcing_start_steps, 0), self.hp.teacher_forcing_steps)

        # run the current model (teacher forcing )
        post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = self.net(src, src_len, trg_mel, trg_len, spkrs, langs, 1.0)
        return post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output

    def parse_batch_by_task(self, x, sel_indices):
        """
        select samples from a batch that correspond to a task
        """
        if x is None: return x
        return torch.index_select(x, 0, sel_indices)


    def observe(self, cur_batch, cur_task):

        # compute gradient on previous tasks
        for task_idx in range(self.n_tasks - 1):
            # -1 means exclude current task
            self.zero_grad()
            for batch in self.mem_data:
                bs = batch[0].size(0)
                batch_langs = batch[7]
                # task_idx: an int representing task identifier.
                # e.g. 0 for german, 1 for french, 2 for spanish
                sel_indices = torch.LongTensor(
                         [i for i in range(bs) if batch_langs[i][0][task_idx]==1])
                if not len(sel_indices) > 0:
                    # batch does not have data for target task
                    continue
                batch_task = [self.parse_batch_by_task(x, sel_indices) for x in batch]
                batch_task = list(map(to_gpu, batch_task))
                src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch_task
                post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = self.forward(batch_task)
                # evaluate loss function
                ptloss, _ = self.criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, trg_mel, stop_pred, stop_trg, alignment, spkrs, spkrs_pred, enc_output, None)
                (ptloss/len(self.mem_data)).backward()
            store_grad(self.parameters, self.grads, self.grad_dims, task_idx)


        # now compute the grad on the current minibatch
        self.zero_grad()

        batch = list(map(to_gpu, cur_batch))
        src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch
        post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = self.forward(batch)
        loss, batch_losses = self.criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, trg_mel, stop_pred, stop_trg, alignment, spkrs, spkrs_pred, enc_output, None)
        loss.backward()

        # check if gradient violates constraints
        # copy gradient
        store_grad(self.parameters, self.grads, self.grad_dims, cur_task)
        indx = torch.cuda.LongTensor(list(range(cur_task)))
        dotp = torch.mm(self.grads[:, cur_task].unsqueeze(0),
                        self.grads.index_select(1, indx))
        if (dotp < 0).sum() != 0:
            project2cone2(self.grads[:, cur_task].unsqueeze(1),
                          self.grads.index_select(1, indx), self.margin)
            # copy gradients back
            overwrite_grad(self.parameters, self.grads[:, cur_task],
                           self.grad_dims)

        gradient = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.hp.gradient_clipping)
        self.opt.step()
        return loss, batch_losses, gradient
