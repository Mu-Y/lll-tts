import os
import time
import datetime
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import TextToSpeechDatasetCollection, TextToSpeechCollate, TextToSpeechDataset
from params.params import Params as hp
from utils import audio, text
from modules.tacotron2 import Tacotron, TacotronLoss
from utils.logging import Logger
from utils.samplers import WeightedSampler, BalancedBatchSampler
from utils import lengths_to_mask, to_gpu
import warnings
from ewc import EWC
from gem import GEM
import copy
import pickle
import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)  # do not print Deprecation Warnings


def cos_decay(global_step, decay_steps):
    """Cosine decay function

    Arguments:
        global_step -- current training step
        decay_steps -- number of decay steps
    """
    global_step = min(global_step, decay_steps)
    return 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))


def train(logging_start_epoch, epoch, data, model, criterion, optimizer, ewc=None):
    """Main training procedure.

    Arguments:
        logging_start_epoch -- number of the first epoch to be logged
        epoch -- current epoch
        data -- DataLoader which can provide batches for an epoch
        model -- model to be trained
        criterion -- instance of loss function to be optimized
        optimizer -- instance of optimizer which will be used for parameter updates
    """

    model.train()

    # initialize counters, etc.
    learning_rate = optimizer.param_groups[0]['lr']
    cla = 0
    done, start_time = 0, time.time()

    # loop through epoch batches
    for i, batch in enumerate(data):

        global_step = done + epoch * len(data)
        optimizer.zero_grad()

        # parse batch
        batch = list(map(to_gpu, batch))
        src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch

        # get teacher forcing ratio
        if hp.constant_teacher_forcing: tf = hp.teacher_forcing
        else: tf = cos_decay(max(global_step - hp.teacher_forcing_start_steps, 0), hp.teacher_forcing_steps)

        # run the current model (student)
        post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = model(src, src_len, trg_mel, trg_len, spkrs, langs, tf)


        # evaluate loss function
        post_trg = trg_lin if hp.predict_linear else trg_mel
        classifier = model._reversal_classifier if hp.reversal_classifier else None
        loss, batch_losses = criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, post_trg, stop_pred, stop_trg, alignment,
                                       spkrs, spkrs_pred, enc_output, classifier)


        # evaluate adversarial classifier accuracy, if present
        if hp.reversal_classifier:
            input_mask = lengths_to_mask(src_len)
            trg_spkrs = torch.zeros_like(input_mask, dtype=torch.int64)
            for s in range(hp.speaker_number):
                speaker_mask = (spkrs == s)
                trg_spkrs[speaker_mask] = s
            matches = (trg_spkrs == torch.argmax(torch.nn.functional.softmax(spkrs_pred, dim=-1), dim=-1))
            matches[~input_mask] = False
            cla = torch.sum(matches).item() / torch.sum(input_mask).item()

        # comptute gradients and make a step
        if ewc is not None:
            loss += hp.ewc_importance * ewc.penalty(model)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.gradient_clipping)
        optimizer.step()

        # log training progress
        if epoch >= logging_start_epoch:
            Logger.training(global_step, batch_losses, gradient, learning_rate, time.time() - start_time, cla)

        # update criterion states (params and decay of the loss and so on ...)
        criterion.update_states()

        start_time = time.time()
        done += 1


def train_aux(logging_start_epoch, epoch, data, model, criterion, optimizer, ewc=None, data_aux=None):
    """Main training procedure.

    Arguments:
        logging_start_epoch -- number of the first epoch to be logged
        epoch -- current epoch
        data -- DataLoader which can provide batches for an epoch
        model -- model to be trained
        criterion -- instance of loss function to be optimized
        optimizer -- instance of optimizer which will be used for parameter updates
    """

    model.train()

    # initialize counters, etc.
    learning_rate = optimizer.param_groups[0]['lr']
    cla = 0
    done, start_time = 0, time.time()

    # loop through epoch batches
    # data: cbs batch; data_aux: rrs batch
    for i, (batch_cbs, batch_rrs) in enumerate(zip(data, data_aux)):

        global_step = done + epoch * len(data)
        optimizer.zero_grad()

        ##### run aux batch (rrs batch)
        # parse batch
        batch = list(map(to_gpu, batch_rrs))
        src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch

        # get teacher forcing ratio
        if hp.constant_teacher_forcing: tf = hp.teacher_forcing
        else: tf = cos_decay(max(global_step - hp.teacher_forcing_start_steps, 0), hp.teacher_forcing_steps)

        # run the current model (student)
        post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = model(src, src_len, trg_mel, trg_len, spkrs, langs, tf, is_rrs=True)


        # evaluate loss function
        post_trg = trg_lin if hp.predict_linear else trg_mel
        classifier = model._reversal_classifier if hp.reversal_classifier else None
        loss_rrs, batch_losses_rrs = criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, post_trg, stop_pred, stop_trg, alignment,
                                       spkrs, spkrs_pred, enc_output, classifier)
        (hp.rrs_importance*loss_rrs).backward()

        ##### run primary batch (cbs batch)
        # parse batch
        batch = list(map(to_gpu, batch_cbs))
        src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch
        # run the current model (student)
        post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = model(src, src_len, trg_mel, trg_len, spkrs, langs, tf, is_rrs=False)


        # evaluate loss function
        post_trg = trg_lin if hp.predict_linear else trg_mel
        classifier = model._reversal_classifier if hp.reversal_classifier else None
        loss, batch_losses = criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, post_trg, stop_pred, stop_trg, alignment,
                                       spkrs, spkrs_pred, enc_output, classifier)
        (hp.cbs_importance*loss).backward()

        # evaluate adversarial classifier accuracy, if present
        if hp.reversal_classifier:
            input_mask = lengths_to_mask(src_len)
            trg_spkrs = torch.zeros_like(input_mask, dtype=torch.int64)
            for s in range(hp.speaker_number):
                speaker_mask = (spkrs == s)
                trg_spkrs[speaker_mask] = s
            matches = (trg_spkrs == torch.argmax(torch.nn.functional.softmax(spkrs_pred, dim=-1), dim=-1))
            matches[~input_mask] = False
            cla = torch.sum(matches).item() / torch.sum(input_mask).item()

        # comptute gradients and make a step
        if ewc is not None:
            loss += hp.ewc_importance * ewc.penalty(model)
        # loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.gradient_clipping)
        optimizer.step()

        # log training progress
        if epoch >= logging_start_epoch:
            Logger.training(global_step, batch_losses, gradient, learning_rate, time.time() - start_time, cla)

        # update criterion states (params and decay of the loss and so on ...)
        criterion.update_states()

        start_time = time.time()
        done += 1

def train_gem(logging_start_epoch, epoch, data, gem, cur_task_id):
    """Main training procedure.

    Arguments:
        logging_start_epoch -- number of the first epoch to be logged
        epoch -- current epoch
        data -- DataLoader which can provide batches for an epoch
        gem -- gem trainer
    """

    gem.train()
    gem.net.train()

    # initialize counters, etc.
    learning_rate = gem.opt.param_groups[0]['lr']
    cla = 0
    done, start_time = 0, time.time()

    # loop through epoch batches
    for i, batch in enumerate(data):

        global_step = done + epoch * len(data)
        gem.opt.zero_grad()

        _, batch_losses, gradient = gem.observe(batch, cur_task_id)


        # evaluate adversarial classifier accuracy, if present
        if hp.reversal_classifier:
            input_mask = lengths_to_mask(src_len)
            trg_spkrs = torch.zeros_like(input_mask, dtype=torch.int64)
            for s in range(hp.speaker_number):
                speaker_mask = (spkrs == s)
                trg_spkrs[speaker_mask] = s
            matches = (trg_spkrs == torch.argmax(torch.nn.functional.softmax(spkrs_pred, dim=-1), dim=-1))
            matches[~input_mask] = False
            cla = torch.sum(matches).item() / torch.sum(input_mask).item()

        # log training progress
        if epoch >= logging_start_epoch:
            Logger.training(global_step, batch_losses, gradient, learning_rate, time.time() - start_time, cla)

        # update criterion states (params and decay of the loss and so on ...)
        gem.criterion.update_states()

        start_time = time.time()
        done += 1

def evaluate(epoch, data, model, criterion, eval_loaders=None):
    """Main evaluation procedure.

    Arguments:
        epoch -- current epoch
        data -- DataLoader which can provide validation batches
        model -- model to be evaluated
        criterion -- instance of loss function to measure performance
    """

    model.eval()

    # initialize counters, etc.
    mcd, mcd_count = 0, 0
    cla, cla_count = 0, 0
    eval_losses = {}

    # loop through epoch batches
    with torch.no_grad():
        for i, batch in enumerate(data):

            # parse batch
            batch = list(map(to_gpu, batch))
            src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch

            # run the model (twice, with and without teacher forcing)
            post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = model(src, src_len, trg_mel, trg_len, spkrs, langs, 1.0)
            post_pred_0, _, stop_pred_0, alignment_0, _, _ = model(src, src_len, trg_mel, trg_len, spkrs, langs, 0.0)
            stop_pred_probs = torch.sigmoid(stop_pred_0)

            # evaluate loss function
            post_trg = trg_lin if hp.predict_linear else trg_mel
            classifier = model._reversal_classifier if hp.reversal_classifier else None
            loss, batch_losses = criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, post_trg, stop_pred, stop_trg, alignment,
                                           spkrs, spkrs_pred, enc_output, classifier)

            # compute mel cepstral distorsion
            for j, (gen, ref, stop) in enumerate(zip(post_pred_0, trg_mel, stop_pred_probs)):
                stop_idxes = np.where(stop.cpu().numpy() > 0.5)[0]
                stop_idx = min(np.min(stop_idxes) + hp.stop_frames, gen.size()[1]) if len(stop_idxes) > 0 else gen.size()[1]
                gen = gen[:, :stop_idx].data.cpu().numpy()
                ref = ref[:, :trg_len[j]].data.cpu().numpy()
                if hp.normalize_spectrogram:
                    gen = audio.denormalize_spectrogram(gen, not hp.predict_linear)
                    ref = audio.denormalize_spectrogram(ref, True)
                if hp.predict_linear: gen = audio.linear_to_mel(gen)
                mcd = (mcd_count * mcd + audio.mel_cepstral_distorision(gen, ref, 'dtw')) / (mcd_count+1)
                mcd_count += 1

            # compute adversarial classifier accuracy
            if hp.reversal_classifier:
                input_mask = lengths_to_mask(src_len)
                trg_spkrs = torch.zeros_like(input_mask, dtype=torch.int64)
                for s in range(hp.speaker_number):
                    speaker_mask = (spkrs == s)
                    trg_spkrs[speaker_mask] = s
                matches = (trg_spkrs == torch.argmax(torch.nn.functional.softmax(spkrs_pred, dim=-1), dim=-1))
                matches[~input_mask] = False
                cla = (cla_count * cla + torch.sum(matches).item() / torch.sum(input_mask).item()) / (cla_count+1)
                cla_count += 1

            # add batch losses to epoch losses
            for k, v in batch_losses.items():
                eval_losses[k] = v + eval_losses[k] if k in eval_losses else v

    # normalize loss per batch
    for k in eval_losses.keys():
        eval_losses[k] /= len(data)

    # log evaluation
    Logger.evaluation(epoch+1, eval_losses, mcd, src_len, trg_len, src, post_trg, post_pred, post_pred_0, stop_pred_probs, stop_trg, alignment_0, cla)

    if eval_loaders is not None:
        for eval_lang, eval_loader in eval_loaders:
            mcd_old_tasks, mcd_count_old_tasks = 0., 0.
            # loop through epoch batches
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):

                    # parse batch
                    batch = list(map(to_gpu, batch))
                    src_old, src_len_old, trg_mel_old, trg_lin_old, trg_len_old, stop_trg_old, spkrs_old, langs_old = batch

                    # run the model (without teacher forcing, computing mcd only)
                    post_pred_0_old, _, stop_pred_0_old, _, _, _ = model(src_old, src_len_old, trg_mel_old, trg_len_old, spkrs_old, langs_old, 0.0)
                    stop_pred_probs = torch.sigmoid(stop_pred_0_old)


                    # compute mel cepstral distorsion
                    for j, (gen, ref, stop) in enumerate(zip(post_pred_0_old, trg_mel_old, stop_pred_probs)):
                        stop_idxes = np.where(stop.cpu().numpy() > 0.5)[0]
                        stop_idx = min(np.min(stop_idxes) + hp.stop_frames, gen.size()[1]) if len(stop_idxes) > 0 else gen.size()[1]
                        gen = gen[:, :stop_idx].data.cpu().numpy()
                        ref = ref[:, :trg_len_old[j]].data.cpu().numpy()
                        if hp.normalize_spectrogram:
                            gen = audio.denormalize_spectrogram(gen, not hp.predict_linear)
                            ref = audio.denormalize_spectrogram(ref, True)
                        if hp.predict_linear: gen = audio.linear_to_mel(gen)
                        mcd_old_tasks = (mcd_count_old_tasks * mcd_old_tasks + audio.mel_cepstral_distorision(gen, ref, 'dtw')) / (mcd_count_old_tasks+1)
                        mcd_count_old_tasks += 1
            # add per-lang mcd to logger
            Logger._sw.add_scalar(f'Eval/mcd_{eval_lang}', mcd_old_tasks, epoch+1)


    return sum(eval_losses.values())


def compute_mcd_on_data_loader(data, model, mel_mean=None, mel_var=None):
    """Main evaluation procedure.

    Arguments:
        epoch -- current epoch
        data -- DataLoader which can provide validation batches
        model -- model to be evaluated
        criterion -- instance of loss function to measure performance
    """

    model.eval()

    # For any given eval langauge, should provide the corresponding cached mel mean and var
    # otherwise the model will use the mel mean and var that it was trained on
    if mel_mean is not None and mel_var is not None:
        hp.mel_normalize_mean = mel_mean
        hp.mel_normalize_variance = mel_var

    # initialize counters, etc.
    mcd, mcd_count = 0, 0

    # loop through epoch batches
    with torch.no_grad():
        for i, batch in enumerate(data):

            # parse batch
            if torch.cuda.is_available():
                batch = list(map(to_gpu, batch))
            src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch

            # run the model (only once, without teacher forcing)
            # post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = model(src, src_len, trg_mel, trg_len, spkrs, langs, 1.0)
            post_pred_0, _, stop_pred_0, alignment_0, _, _ = model(src, src_len, trg_mel, trg_len, spkrs, langs, 0.0)
            stop_pred_probs = torch.sigmoid(stop_pred_0)


            # compute mel cepstral distorsion
            for j, (gen, ref, stop) in enumerate(zip(post_pred_0, trg_mel, stop_pred_probs)):
                stop_idxes = np.where(stop.cpu().numpy() > 0.5)[0]
                stop_idx = min(np.min(stop_idxes) + hp.stop_frames, gen.size()[1]) if len(stop_idxes) > 0 else gen.size()[1]
                gen = gen[:, :stop_idx].data.cpu().numpy()
                ref = ref[:, :trg_len[j]].data.cpu().numpy()
                if hp.normalize_spectrogram:
                    gen = audio.denormalize_spectrogram(gen, not hp.predict_linear)
                    ref = audio.denormalize_spectrogram(ref, True)
                if hp.predict_linear: gen = audio.linear_to_mel(gen)
                mcd = (mcd_count * mcd + audio.mel_cepstral_distorision(gen, ref, 'dtw')) / (mcd_count+1)
                mcd_count += 1

    return mcd


class DataParallelPassthrough(torch.nn.DataParallel):
    """Simple wrapper around DataParallel."""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


if __name__ == '__main__':
    import argparse
    import os
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=".", help="Base directory of the project.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Name of the initial checkpoint.")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints", help="Base directory of checkpoints.")
    parser.add_argument("--warm_start", action="store_true", help="if set, start training using the optimizer state in checkpoint; if not set, start training with newly optimizer state but the pretrained model weights")
    parser.add_argument("--data_root", type=str, default="data", help="Base directory of datasets.")
    parser.add_argument("--flush_seconds", type=int, default=60, help="How often to flush pending summaries to tensorboard.")
    parser.add_argument('--hyper_parameters', type=str, default=None, help="Name of the hyperparameters file.")
    parser.add_argument('--logging_start', type=int, default=1, help="First epoch to be logged")
    parser.add_argument('--max_gpus', type=int, default=2, help="Maximal number of GPUs of the local machine to use.")
    parser.add_argument('--loader_workers', type=int, default=16, help="Number of subprocesses to use for data loading.")
    args = parser.parse_args()

    # set up seeds and the target torch device
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # prepare directory for checkpoints
    checkpoint_dir = os.path.join(args.save_dir, args.checkpoint_root)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # load checkpoint (dict) with saved hyper-parameters (let some of them be overwritten because of fine-tuning)
    if args.checkpoint:
        checkpoint = os.path.join(args.checkpoint)
        checkpoint_state = torch.load(checkpoint, map_location='cpu')
        # hp.load_state_dict(checkpoint_state['parameters'])

    # load hyperparameters
    if args.hyper_parameters is not None:
        hp_path = os.path.join('params', args.hyper_parameters)
        hp.load(hp_path)


    # initialize logger
    log_dir = os.path.join(args.save_dir, "logs", f'{hp.version}-{datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")}')
    Logger.initialize(log_dir, args.flush_seconds)



    ## always use the mean and std from all10 languages
    with open("stats_per_lang_w-all10.pkl", "rb") as f:
        stats = pickle.load(f)
    hp.mel_normalize_mean = stats["all10"]["mel_normalize_mean"]
    hp.mel_normalize_variance = stats["all10"]["mel_normalize_variance"]

    ewc = None
    prev_model, criterion_kd = None, None
    if hp.use_replay:
        prev_samples= TextToSpeechDataset(
            os.path.join(args.data_root, hp.dataset, f"train_{hp.languages[0]}_w-ipa.txt" if hp.use_phonemes else f"train_{hp.languages[0]}.txt"),
            os.path.join(args.data_root, hp.dataset),
            sample_size=hp.replay_size)

    initial_epoch=0
    for task_idx, train_lang in enumerate(hp.training_langs, 1):

        # check directory
        if not os.path.exists(os.path.join(checkpoint_dir, train_lang)):
            os.makedirs(os.path.join(checkpoint_dir, train_lang))
        if not os.path.exists(os.path.join(args.save_dir, "logs")):
            os.makedirs(os.path.join(args.save_dir, "logs"))


        # load dataset
        dataset = TextToSpeechDatasetCollection(os.path.join(args.data_root, hp.dataset),
                f"train_{train_lang}_w-ipa.txt" if hp.use_phonemes else f"train_{train_lang}.txt",
                f"val_{train_lang}_w-ipa.txt" if hp.use_phonemes else f"val_{train_lang}.txt")


        if hp.use_replay and (not hp.use_gem):
            dataset.train.concat_dataset(prev_samples)
            if hp.weighted_sampling:
                sampler = WeightedSampler(dataset.train)
                train_data = DataLoader(dataset.train, batch_size=hp.batch_size, drop_last=True,
                                shuffle=False, sampler=WeightedSampler(dataset.train),
                                collate_fn=TextToSpeechCollate(True), num_workers=args.loader_workers)
            elif hp.dual_sampling:
                train_data_rrs = DataLoader(dataset.train, batch_size=hp.batch_size, drop_last=True,
                                shuffle=True, sampler=None,
                                collate_fn=TextToSpeechCollate(True), num_workers=args.loader_workers)
                train_data = DataLoader(dataset.train,
                                batch_sampler=BalancedBatchSampler(dataset.train, hp.batch_size,
                                                    len(train_data_rrs)*hp.batch_size, shuffle=True),
                                collate_fn=TextToSpeechCollate(True), num_workers=args.loader_workers)
                assert len(train_data_rrs) == len(train_data)
            else:
                train_data = DataLoader(dataset.train, batch_size=hp.batch_size, drop_last=True,
                                shuffle=True, sampler=None,
                                collate_fn=TextToSpeechCollate(True), num_workers=args.loader_workers)


        if hp.use_replay and hp.use_gem:
            ## For each new task, the data loader should only have 1 lang (current lang)
            assert dataset.train.get_num_languages() == 1, print("Current task data loader has >1 langs")
            train_data = DataLoader(dataset.train, batch_size=hp.batch_size, drop_last=True,
                                shuffle=True, sampler=None,
                        collate_fn=TextToSpeechCollate(True), num_workers=args.loader_workers)
            prev_data = DataLoader(prev_samples, batch_size=hp.batch_size, drop_last=False,
                                shuffle=True, sampler=None,
                        collate_fn=TextToSpeechCollate(True), num_workers=args.loader_workers)

        if not hp.use_replay:
            train_data = DataLoader(dataset.train, batch_size=hp.batch_size, drop_last=True,
                                shuffle=True, sampler=None,
                        collate_fn=TextToSpeechCollate(True), num_workers=args.loader_workers)

        eval_data = DataLoader(dataset.dev, batch_size=hp.batch_size, drop_last=False, shuffle=False,
                               collate_fn=TextToSpeechCollate(True), num_workers=args.loader_workers)


        # find out number of unique speakers and languages
        hp.speaker_number = 0 if not hp.multi_speaker else dataset.train.get_num_speakers()
        hp.language_number = 0 if not hp.multi_language else len(hp.languages)
        # save all found speakers to hyper parameters
        if hp.multi_speaker and not args.checkpoint:
            hp.unique_speakers = dataset.train.unique_speakers


        # instantiate model
        if torch.cuda.is_available():
            model = Tacotron().cuda()
            if hp.parallelization and args.max_gpus > 1 and torch.cuda.device_count() > 1:
                model = DataParallelPassthrough(model, device_ids=list(range(args.max_gpus)))
        else: model = Tacotron()


        # instantiate optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
        if hp.encoder_optimizer:
            encoder_params = list(model._encoder.parameters())
            other_params = list(model._decoder.parameters()) + list(model._postnet.parameters()) + list(model._prenet.parameters()) + \
                           list(model._embedding.parameters()) + list(model._attention.parameters())
            if hp.reversal_classifier:
                other_params += list(model._reversal_classifier.parameters())
            optimizer = torch.optim.Adam([
                {'params': other_params},
                {'params': encoder_params, 'lr': hp.learning_rate_encoder}
            ], lr=hp.learning_rate, weight_decay=hp.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hp.learning_rate_decay_each, gamma=hp.learning_rate_decay)
        criterion = TacotronLoss(hp.guided_attention_steps, hp.guided_attention_toleration, hp.guided_attention_gain)


        # load model weights, fisher, and optimizer, scheduler states from checkpoint state dictionary
        # initial_epoch = 0
        if args.checkpoint:
            checkpoint_state = torch.load(args.checkpoint, map_location='cpu')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_state['model'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            if args.warm_start:
                # other states from checkpoint -- optimizer, scheduler, loss, epoch step num
                # for guided_attn
                initial_epoch = checkpoint_state['epoch'] + 1
                optimizer.load_state_dict(checkpoint_state['optimizer'])
                scheduler.load_state_dict(checkpoint_state['scheduler'])
                criterion.load_state_dict(checkpoint_state['criterion'])
            print("model loaded from {}".format(args.checkpoint))
            if hp.use_ewc:
                ewc = EWC(model, criterion)
                ewc.load_fisher(checkpoint_state['fisher'])
                ############################################
                # dataset_old = TextToSpeechDatasetCollection(os.path.join(args.data_root, hp.dataset),
                #     f"train_german_w-ipa.txt" if hp.use_phonemes else f"train_german.txt",
                #     f"val_german_w-ipa.txt" if hp.use_phonemes else f"val_german.txt")
                # train_data_old = DataLoader(dataset_old.train, batch_size=hp.batch_size,
                #         drop_last=True, shuffle=False,
                #         sampler=None, collate_fn=TextToSpeechCollate(True),
                #         num_workers=args.loader_workers)
                # ewc.update_fisher(train_data_old)
                # state_dict = {
                #     'epoch': 99,
                #     'model': model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'scheduler': scheduler.state_dict(),
                #     'parameters': hp.state_dict(),
                #     'criterion': criterion.state_dict(),
                #     'fisher': ewc.get_fisher()
                # }
                # torch.save(state_dict, "{}-fisher".format(args.checkpoint))
                # print("model saved to {}-fisher".format(args.checkpoint))
                ###########################################




        ## prepare eval data for each lang. For purpose of eval forgetting on each previous lang
        eval_loaders = []
        eval_langs = hp.languages[:hp.languages.index(train_lang)]
        for lang in eval_langs:
            # load dataset
            lang_dataset = TextToSpeechDatasetCollection(os.path.join(args.data_root, hp.dataset),
                                                    training_file=None,
            validation_file=f"val_{lang}_w-ipa.txt" if hp.use_phonemes else f"val_{lang}.txt")
            lang_eval_loader = DataLoader(lang_dataset.dev, batch_size=hp.batch_size, drop_last=False,
                                          shuffle=False, collate_fn=TextToSpeechCollate(True),
                                          num_workers=args.loader_workers)
            eval_loaders.append((lang, lang_eval_loader))

        if hp.use_gem:
            n_tasks = prev_samples.get_num_languages() + 1 # +1 to include the current task
            gem = GEM(model, criterion, optimizer, prev_data, n_tasks, hp)
            cur_task_id = n_tasks - 1  # cur_task_id is always the last task

        # training loop
        best_eval = float('inf')
        for epoch in range(initial_epoch, initial_epoch + hp.epochs):
            if hp.use_replay and hp.dual_sampling:
                train_aux(args.logging_start, epoch, train_data, model, criterion, optimizer, ewc, train_data_rrs)
            elif hp.use_replay and hp.use_gem:
                train_gem(args.logging_start, epoch, train_data, gem, cur_task_id)
            else:
                train(args.logging_start, epoch, train_data, model, criterion, optimizer, ewc)
            # if hp.learning_rate_decay_start - hp.learning_rate_decay_each < epoch * len(train_data):
            scheduler.step()
            eval_loss = evaluate(epoch, eval_data, model, criterion, eval_loaders)
            print("Epoch: {}, Eval_loss: {}".format(epoch, eval_loss))
            if (epoch + 1) % hp.checkpoint_each_epochs == 0:
                # save checkpoint together with hyper-parameters, optimizer and scheduler states
                checkpoint_file = f'{checkpoint_dir}/{train_lang}/{hp.version}_loss-{epoch}-{eval_loss:2.3f}'
                state_dict = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'parameters': hp.state_dict(),
                    'criterion': criterion.state_dict()
                }
                torch.save(state_dict, checkpoint_file)
                print("Saved model to {}".format(checkpoint_file))

        # after training on the current task, update the ewc fisher
        if hp.use_ewc:
            ewc = EWC(model, criterion)
            ewc.update_fisher(train_data)
            state_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'parameters': hp.state_dict(),
                'criterion': criterion.state_dict(),
                'fisher': ewc.get_fisher()
            }
            torch.save(state_dict, "{}-fisher".format(checkpoint_file))
            print("model saved to {}-fisher".format(checkpoint_file))
        # update the checkpoint to be used in the next task
        if hp.use_ewc:
            args.checkpoint = "{}-fisher".format(checkpoint_file)
        else:
            args.checkpoint = checkpoint_file

        # update previous samples with new task data
        if hp.use_replay:
            n_keep_items = int(hp.replay_size * task_idx // (task_idx+1))
            n_new_items = int(hp.replay_size * 1 // (task_idx+1))
            prev_samples = prev_samples.sample_n_items(n_keep_items)
            cur_dataset= TextToSpeechDataset(
                os.path.join(args.data_root, hp.dataset, f"train_{train_lang}_w-ipa.txt" if hp.use_phonemes else f"train_{train_lang}.txt"),
                os.path.join(args.data_root, hp.dataset))
            prev_samples_add = cur_dataset.sample_n_items(n_new_items)
            prev_samples.concat_dataset(prev_samples_add)

        # re-initialize epoch num
        initial_epoch += hp.epochs

