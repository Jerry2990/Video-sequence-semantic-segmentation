import argparse
from util.parser import str2bool
# Parser
parser = argparse.ArgumentParser(description='Training End2End Network')

# Basic configuration
parser.add_argument('--device', type=str, default="gpu", help="Device to run. Options: gpu/cpu")
parser.add_argument('--local_rank', type=int, help='Local process ID#.', default=0)
parser.add_argument('--freeze-bn', type=str2bool, default='0', help="(Fixed)")
parser.add_argument('--num-worker', type=int, default=4)

# Hyper Parameters
parser.add_argument('--lr', type=float, help="Learning rate", default=0.002)
parser.add_argument('--guide-lr', type=float, help="Learning rate for guiding network", default=0.02)
parser.add_argument('--lr-gamma', type=float, default=0.992)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--param-ce', type=float, help='Loss parameter -- (Fixed)', default=10)
parser.add_argument('--param-ce-distil', type=float, help='dataLoss parameter -- (Fixed)', default=10)

# Model
parser.add_argument('--intra', type=str2bool, help="(Fixed)", default='1')
parser.add_argument('--train-filter', type=str2bool, help="train 'filter_kernel_s' or not(Fixed)", default='0')

# Training
parser.add_argument('--distil', type=str2bool, default='1', help="enable distillation loss.")
parser.add_argument('--optim', type=str, help="Set model optimizer adam/sgd", default="sgd")
parser.add_argument('--step-scheduler', type=int, default=100, help="step size of StepLR")
parser.add_argument('--load-flow', type=str2bool, default='1', help="")
parser.add_argument('--finetune-all',type=str2bool, help="Finetune all or just the last two. (Fixed)", default='0')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1) (Fixed)')
parser.add_argument('--epoch', type=int, help='Training interations', default=100)
parser.add_argument('--crop-height', type=int, default=768, help='crop height of data augmentation')
parser.add_argument('--crop-width', type=int, default=1536, help='crop width of data augmentation')
parser.add_argument('--use-crop', type=str2bool, default='1', help='to use crop or not')
parser.add_argument('--early-stop', type=str2bool, help="run 5 iters each epoch(for testing) ", default=0)

# Dataset
parser.add_argument('--dataset',type=str, help="Options: cityscapes_2k/camvid", default="cityscapes_2k")
parser.add_argument('--multi-step', nargs='+', help="(Fixed)", type=int, default=[1,2,3,4])#[1,2,3,4])
parser.add_argument('--eval-multi-step', nargs='+', help="the number of algorigtm step", type=int, default=[1,2,3,4])#[1,2,3,4])
parser.add_argument('--eval-single',type=str2bool, help="Evaluate inference on step 0.(Fixed)", default='0')

# Segmentation Network
parser.add_argument('--segnet', type=str, help="Segmentation Network. Options: bisenet/swiftnet", default="swiftnet")
parser.add_argument('--optical-flow-network', type=str, default='light', help='Optical flow network. Options: light/flownet')
parser.add_argument('--optical-flow-lr', type=float, help="learning rate of the optical flow", default=0.02)
parser.add_argument('--train-optical-flow', type=str2bool, default=1, help='1 for training, 0 for evaluating')

# Evaluation
parser.add_argument('--evaluate', type=str2bool, default='0', help="set to 1 if evaluate")
parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (default: 1). (Fixed)')
parser.add_argument('--eval-scale', type=float, default=0.75, help="scale of original input image resolution")
parser.add_argument('--checkname', type=str, default='swnet-seq-test', help="Name of this run")
parser.add_argument('--log-folder', type=str, help="Log folder", default='./logs')
parser.add_argument('--reset-optimizer', type=str2bool, help="(Fixed)", default='1')
parser.add_argument('--reset-best-pred', type=str2bool, help="reset miou(initialize only): 1 , don't reset miou(continue model for finetuning): 0", default='1')
parser.add_argument('--resume', type=str2bool, default='1', help="Initialize model")
parser.add_argument('--visualization', type=str2bool, default=0, help='to visualize predictions')

# Inference
parser.add_argument('--inference', type=str2bool, default='0', help="only infer or not")

# Save results
parser.add_argument('--save-res', type=str2bool, default='1', help="save eval results or not")
parser.add_argument('--save-seq-res', type=str2bool, default='1', help="save eval sequence results between two key frames or not")
parser.add_argument('--ResFolderName', type=str, default='l3_wedge_test', help="Name of pred results folder")

args = parser.parse_args()

args.resume = args.inference or args.evaluate or args.resume
args.save_res = (args.inference) or (args.evaluate and args.save_res)
args.save_seq_res = (args.inference or args.evaluate) and args.save_seq_res

args.multi_step = [1,2]
args.eval_multi_step = [0,1,2]

from trainer.gsvnet_trainer import GSVNet_Trainer
trainer = GSVNet_Trainer(args)
if args.inference:
    trainer.inference(args)
else:
    trainer.run()