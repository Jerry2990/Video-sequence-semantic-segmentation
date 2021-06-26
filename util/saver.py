import os
import shutil
import torch
import glob
import sys
from PIL import Image
import  numpy as np
import scipy.misc as misc
import imageio
import torch
from config.cityscapes import cityscapes_config
from config.camvid import camvid_config
import torch.nn.functional as F
class Saver (object):
    def __init__(self, args, name, directory=None):
        self.args = args
        if directory:
            self.directory = directory
        else:
            self.directory = os.path.join(args.log_folder, 'run', args.dataset, name)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self, dict):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        for key, val in dict.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()

def save_res(final_pred, img_names, path, dataset_name, interval):
    batch = len(final_pred)
    assert batch == len(final_pred)
    assert batch == len(img_names)


    final_pred = final_pred.cpu().data.numpy()
    for idx in range(batch):
        pred = final_pred[idx]
        # pred = pred.squeeze(0)
        pred = get_color_pallete(pred, dataset_name)
        pred = np.asarray(pred.convert('RGB'))
        imageio.imwrite(os.path.join(path, f"{img_names[idx]}_pred_k{interval}.png"), pred)


def save_seq_res(pred_sequence,img_names,path):
    l_frame = len(pred_sequence)
    batch = len(pred_sequence[0])
    assert l_frame == len(img_names)
    for frame_idx in range(l_frame):
        pred_sequence[frame_idx] = parse_seg(pred_sequence[frame_idx])
    sequence = [pred_sequence]
    res_sequence = [] # 2 batch
    for idx in range(batch):
        cur_batch = [] # 3 seq
        for jdx in range(len(sequence)):
            cur_batch.append(torch.cat([batches[idx] for batches in sequence[jdx]],dim=1))

        res_sequence.append(torch.cat(cur_batch,dim=0))

    img_names = img_names[-1]
    for idx, img in enumerate(res_sequence):
        imageio.imwrite(os.path.join(path, f"{img_names[idx]}_seq_k{l_frame}.png"), img)

    a=1




def parse_seg(seg_batch):
    seg_batch = torch.argmax(F.interpolate(seg_batch, size=(1024, 2048), mode='bilinear'), dim=1).cpu().data.numpy()
    results = []
    for i in range(len(seg_batch)):
        res = get_color_pallete(seg_batch[i], "cityscapes_2k")
        res = np.asarray(res.convert('RGB'))
        results.append(torch.tensor(res, dtype=torch.float32))
        # results.append(res)

    return results


def parse_img(img_batch):
    u = [0.406, 0.456, 0.485];
    t = [0.225, 0.224, 0.229]
    u = [72.3, 82.90, 73.15]
    t = [47.73, 48.49, 47.67]
    _std = np.array(u).reshape((1, 1, 3))
    _mean = np.array(t).reshape((1, 1, 3))
    img_batch = img_batch.cpu().data.numpy()
    results = []
    for j in range(len(img_batch)):
        img = img_batch[j].transpose(1, 2, 0)
        img = img*_mean+_std
        results.append(torch.tensor(img[:,:,::-1].copy(),dtype=torch.float32))
        # results.append(img)

    return results





def get_color_pallete(npimg, dataset='cityscapes_2k'):
    """Visualize image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    dataset : str,
        The dataset that model pretrained on. ('cityscapes_2k', 'camvid')
    Returns
    -------
    out_img : PIL.Image
        Image with color pallete
    """

    # put colormap
    if dataset == 'cityscapes_2k':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(cityscapes_config.pallete)
        return out_img
    elif dataset == 'camvid':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(camvid_config.pallete)
        return out_img
    return out_img
