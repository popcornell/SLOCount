import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from .assets import Binarymetrics
import matplotlib.pyplot as plt

def validate(hp,  model, testloader, writer, step, name="dev"):


    vad_scores = Binarymetrics.BinaryMeter()  # activity scores
    vod_scores = Binarymetrics.BinaryMeter()  # overlap scores
    count_scores = Binarymetrics.MultiMeter()  # Countnet scores

    model.eval()

    tot_loss = 0

    with torch.no_grad():

        with tqdm(testloader) as t:
            t.set_description("Evaluating on Dev set.")
            for count, batch in enumerate(testloader):

                features, labels = batch
                features = features.cuda()

                labels = labels.cuda()

                preds = model(features)

                loss = criterion(preds, labels)

                VADpreds = torch.sum(torch.exp(preds[:, 1:5, :]), dim=1).unsqueeze(1)
                VADlabels = torch.sum(labels[:, 1:5, :], dim=1).unsqueeze(1)
                vad_scores.update(VADpreds, VADlabels)

                VODpreds = torch.sum(torch.exp(preds[:, 2:5, :]), dim=1).unsqueeze(1)
                VODlabels = torch.sum(labels[:, 2:5, :], dim=1).unsqueeze(1)
                vod_scores.update(VODpreds, VODlabels)

                '''

                for i in range(VODpreds.shape[0]): # iterate over batches
                    import soundfile as sf

                    plt.plot(VODpreds[i, 0, :].cpu().detach().numpy())
                    plt.plot(VODlabels[i, 0, :].cpu().detach().numpy())
                    sf.write("/home/sam/Desktop/temp.wav", audio.detach().numpy()[i, :, 0], 16000)
                    plt.show()
                    
                '''


                count_scores.update(torch.argmax(torch.exp(preds), 1).unsqueeze(1),
                                    torch.argmax(labels, 1).unsqueeze(1), )

                tot_loss += loss.item()

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
                              vod_recall=vod_recall, vod_matt=vod_matt, vod_f1=vod_f1,
                              count_miss=count_miss, count_fa=count_fa, count_prec=count_precision,
                              count_recall=count_recall, count_matt=count_matt, count_f1=count_f1
                              )
                t.update()
                t.update()

        writer.log_metrics("{}_vad".format(name), loss, vad_fa, vad_miss, vad_recall, vad_precision, vad_f1,
                           vad_matt, vad_tp, vad_tn, vad_fp, vad_fn, step)
        writer.log_metrics("{}_vod".format(name), loss, vod_fa, vod_miss, vod_recall, vod_precision, vod_f1,
                           vod_matt, vod_tp, vod_tn, vod_fp, vod_fn, step)
        writer.log_metrics("{}_count".format(name), loss, count_fa, count_miss, count_recall, count_precision, count_f1,
                           count_matt, count_tp, count_tn, count_fp, count_fn, step)

        return tot_loss / (count+1)


def criterion(preds, labels):
    # preds = predicted log-probabilities
    # labels = ground truth probabilities
    return -labels.mul(preds).sum(1).mean()