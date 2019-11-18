import os
import math
import numpy as np
import torch
import traceback
from .assets import Binarymetrics
from .eval import validate
from .architecture import get_SLOCountNet
from torch.nn.parallel.data_parallel import DataParallel
from tqdm import tqdm


def train(args, pt_dir, chkpt_path, trainloader, devloader,  writer, logger, hp, hp_str):

    model = get_SLOCountNet(hp).cuda()

    print("FOV: {}", model.get_fov(hp.features.n_fft))
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("N_parameters : {}".format(params))
    model = DataParallel(model)

    if hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)

    epoch = 0
    best_loss = np.inf

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['step']

        # will use new given hparams.
        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")

    try:

        for epoch in range(epoch, hp.train.n_epochs):

            vad_scores = Binarymetrics.BinaryMeter()  # activity scores
            vod_scores = Binarymetrics.BinaryMeter()  # overlap scores
            count_scores = Binarymetrics.MultiMeter()  # Countnet scores

            model.train()
            tot_loss = 0

            with tqdm(trainloader) as t:
                t.set_description("Epoch: {}".format(epoch))

                for count, batch in enumerate(trainloader):

                    features, labels = batch
                    features = features.cuda()
                    labels = labels.cuda()

                    preds = model(features)

                    loss = criterion(preds, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # compute proper metrics for VAD
                    loss = loss.item()

                    if loss > 1e8 or math.isnan(loss):  # check if exploded
                        logger.error("Loss exploded to %.02f at step %d!" % (loss, epoch))
                        raise Exception("Loss exploded")

                    VADpreds = torch.sum(torch.exp(preds[:, 1:5, :]), dim=1).unsqueeze(1)
                    VADlabels = torch.sum(labels[:, 1:5, :], dim=1).unsqueeze(1)
                    vad_scores.update(VADpreds, VADlabels)

                    VODpreds = torch.sum(torch.exp(preds[:, 2:5, :]), dim=1).unsqueeze(1)
                    VODlabels = torch.sum(labels[:, 2:5, :], dim=1).unsqueeze(1)
                    vod_scores.update(VODpreds, VODlabels)

                    count_scores.update(torch.argmax(torch.exp(preds), 1).unsqueeze(1),
                                        torch.argmax(labels, 1).unsqueeze(1))

                    tot_loss += loss

                    vad_fa = vad_scores.get_fa().item()
                    vad_miss = vad_scores.get_miss().item()
                    vad_precision = vad_scores.get_precision().item()
                    vad_recall = vad_scores.get_recall().item()
                    vad_matt = vad_scores.get_matt().item()
                    vad_f1 = vad_scores.get_f1().item()
                    vad_tp = vad_scores.tp.item()
                    vad_tn = vad_scores.tn.item()
                    vad_fp = vad_scores.fp.item()
                    vad_fn = vad_scores.fn.item()

                    vod_fa = vod_scores.get_fa().item()
                    vod_miss = vod_scores.get_miss().item()
                    vod_precision = vod_scores.get_precision().item()
                    vod_recall = vod_scores.get_recall().item()
                    vod_matt = vod_scores.get_matt().item()
                    vod_f1 = vod_scores.get_f1().item()
                    vod_tp = vod_scores.tp.item()
                    vod_tn = vod_scores.tn.item()
                    vod_fp = vod_scores.fp.item()
                    vod_fn = vod_scores.fn.item()

                    count_fa = count_scores.get_accuracy().item()
                    count_miss = count_scores.get_miss().item()
                    count_precision = count_scores.get_precision().item()
                    count_recall = count_scores.get_recall().item()
                    count_matt = count_scores.get_matt().item()
                    count_f1 = count_scores.get_f1().item()
                    count_tp = count_scores.get_tp().item()
                    count_tn = count_scores.get_tn().item()
                    count_fp = count_scores.get_fp().item()
                    count_fn = count_scores.get_fn().item()



                    t.set_postfix(loss=tot_loss / (count + 1), vad_miss=vad_miss, vad_fa=vad_fa, vad_prec=vad_precision,
                                  vad_recall=vad_recall, vad_matt=vad_matt, vad_f1=vad_f1,
                                  vod_miss=vod_miss, vod_fa=vod_fa, vod_prec=vod_precision,
                                  vod_recall=vod_recall, vod_matt=vod_matt, vod_f1 = vod_f1,
                                  count_miss=count_miss, count_fa=count_fa, count_prec=count_precision,
                                  count_recall=count_recall, count_matt=count_matt, count_f1= count_f1
                                  )
                    t.update()


            writer.log_metrics("train_vad", loss, vad_fa, vad_miss, vad_recall, vad_precision, vad_f1,
                               vad_matt, vad_tp, vad_tn, vad_fp, vad_fn, epoch)
            writer.log_metrics("train_vod", loss, vod_fa, vod_miss, vod_recall, vod_precision, vod_f1,
                               vod_matt,vod_tp, vod_tn, vod_fp, vod_fn,  epoch)
            writer.log_metrics("train_count", loss, count_fa, count_miss, count_recall, count_precision, count_f1,
                               count_matt, count_tp, count_tn, count_fp, count_fn, epoch)
            # end epoch save model and validate it

            val_loss = validate(hp, model, devloader, writer, epoch)


            if hp.train.save_best == 0:
                save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % epoch)
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': epoch,
                    'hp_str': hp_str,
                }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)

            else:
                if val_loss < best_loss:  # save only when best
                    best_loss = val_loss
                    save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % epoch)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': epoch,
                        'hp_str': hp_str,
                    }, save_path)
                logger.info("Saved checkpoint to: %s" % save_path)

        return best_loss

    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()


def criterion(preds, labels):
    # preds = predicted log-probabilities
    # labels = ground truth probabilities
    return -labels.mul(preds).sum(1).mean()
