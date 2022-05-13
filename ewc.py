from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import to_gpu
from params.params import Params as hp
from modules.tacotron2 import Tacotron, TacotronLoss


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module,
                       criterion: nn.Module):

        self.model = model
        self.criterion = criterion
        # self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)


    def _diag_fisher(self, dataset, sample_size=None):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.train()
        # for input in self.dataset:
        #     self.model.zero_grad()
        #     input = variable(input)
        #     output = self.model(input).view(1, -1)
        #     label = output.max(1)[1].view(-1)
        #     loss = F.nll_loss(F.log_softmax(output, dim=1), label)
        #     loss.backward()

        print("Computing Fisher...")
        cnt = 0.
        for i, batch in enumerate(dataset):
            # parse batch
            batch = list(map(to_gpu, batch))
            src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch

            # get teacher forcing ratio
            if hp.constant_teacher_forcing: tf = hp.teacher_forcing
            else: tf = cos_decay(max(global_step - hp.teacher_forcing_start_steps, 0), hp.teacher_forcing_steps)

            # run the model
            post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = self.model(src, src_len, trg_mel, trg_len, spkrs, langs, tf)

            # evaluate loss function
            post_trg = trg_lin if hp.predict_linear else trg_mel
            classifier = model._reversal_classifier if hp.reversal_classifier else None
            loss, batch_losses = self.criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, post_trg, stop_pred, stop_trg, alignment, spkrs, spkrs_pred, enc_output, classifier)
            loss.backward()


            cnt += src.size(0)

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2

            if sample_size:
                if cnt >= sample_size:
                    break
        print("computed Fisher using {} samples".format(cnt))


        precision_matrices = {n: p/cnt for n, p in precision_matrices.items()}  # mean over sampled data
        return precision_matrices

    def update_fisher(self, dataset: DataLoader, sample_size=None):
        self._precision_matrices = self._diag_fisher(dataset, sample_size=sample_size)

    def load_fisher(self, fisher_matrices):
        assert fisher_matrices != {}
        self._precision_matrices = {n:p.cuda() for n, p in fisher_matrices.items()}
        print("fisher loaded.")

    def get_fisher(self):
        return self._precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
