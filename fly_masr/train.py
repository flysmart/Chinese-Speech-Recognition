import numpy as np
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.LAS.decoder import Decoder
from models.LAS.encoder import Encoder
from models.LAS.optimizer import LasOptimizer
from models.LAS.seq2seq import Seq2Seq
from data.data_load import AiShellDataset, pad_collate
from config.conf import device, print_freq, vocab_size, num_workers, sos_id, eos_id
from utils.util import get_logger, save_checkpoint, AverageMeter
import matplotlib
#不显示图片
matplotlib.use('agg')
import matplotlib.pyplot as plt
from data.data_process import data_pre
import os

parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Training "
    "(Transformer framework).")
# Low Frame Rate (stacking and skipping frames)
parser.add_argument('--LFR_m', default=4, type=int,
                    help='Low Frame Rate: number of frames to stack')
parser.add_argument('--LFR_n', default=3, type=int,
                    help='Low Frame Rate: number of frames to skip')
# general
# Network architecture
# encoder
# TODO: automatically infer input dim
parser.add_argument('--einput', default=80, type=int,
                    help='Dim of encoder input')
parser.add_argument('--ehidden', default=256, type=int,
                    help='Size of encoder hidden units')
parser.add_argument('--elayer', default=3, type=int,
                    help='Number of encoder layers.')
parser.add_argument('--edropout', default=0.2, type=float,
                    help='Encoder dropout rate')
parser.add_argument('--ebidirectional', default=True, type=bool,
                    help='Whether use bidirectional encoder')
parser.add_argument('--etype', default='lstm', type=str,
                    help='Type of encoder RNN')
# attention
parser.add_argument('--atype', default='dot', type=str,
                    help='Type of attention (Only support Dot Product now)')
# decoder
parser.add_argument('--dembed', default=512, type=int,
                    help='Size of decoder embedding')
parser.add_argument('--dhidden', default=512, type=int,
                    help='Size of decoder hidden units. Should be encoder '
                         '(2*) hidden size dependding on bidirection')
parser.add_argument('--dlayer', default=1, type=int,
                    help='Number of decoder layers.')
# Training config
parser.add_argument('--epochs', default=10000, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=True, type=bool,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when halving lr but still get'
                         'small improvement')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--batch-size', '-b', default=32, type=int,
                    help='Batch size')
parser.add_argument('--maxlen_in', default=800, type=int, metavar='ML',
                    help='Batch size is reduced if the input sequence length > ML')
parser.add_argument('--maxlen_out', default=150, type=int, metavar='ML',
                    help='Batch size is reduced if the output sequence length > ML')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-2, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=1e-5, type=float,
                    help='weight decay (L2 penalty)')
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')

writer = SummaryWriter()

if os.path.exists("train_losses.npy"):
    train_losses = np.load("train_losses.npy")
    valid_losses = np.load("valid_losses.npy")
    train_losses = train_losses.tolist()
    valid_losses = valid_losses.tolist()
else:
    train_losses = []
    valid_losses = []


def save_and_trace_attention(attn):
    fig = plt.figure(figsize=(12, 6))
    plt.xlabel("encoder step")
    plt.ylabel("decoder step")
    plt.imshow(torch.tensor(attn.squeeze(1), device='cpu'), interpolation='nearest', aspect='auto')
    fig.savefig('alignment.png', bbox_inches='tight')
    plt.close(fig)

def save_and_trace_attention1(attn_distribution):
    batch_size, num_heads, seq_len, input_lengths = attn_distribution.size()
    fig = plt.figure(figsize=(12, 6))
    for i in range(num_heads):
        plt.subplot(2,2,i+1)
        plt.xlabel("encoder step")
        plt.ylabel("decoder step")
        plt.imshow(torch.tensor(attn_distribution[:,i,:,:].squeeze(1), device='cpu'), interpolation='nearest', aspect='auto')
    fig.savefig('alignment.png', bbox_inches='tight')
    plt.close(fig)


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        encoder = Encoder(args.einput * args.LFR_m, args.ehidden, args.elayer,
                          dropout=args.edropout, bidirectional=args.ebidirectional,
                          rnn_type=args.etype)
        decoder = Decoder(vocab_size, args.dembed, sos_id,
                          eos_id, args.dhidden, args.dlayer,
                          bidirectional_encoder=args.ebidirectional)
        model = Seq2Seq(encoder, decoder)
        print(model)
        model.to(device)

        optimizer = LasOptimizer(
            torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
                             eps=1e-09))  # betas：一阶矩阵估计的指数衰减率估算值和二阶矩估计的指数衰减率  eps：以防止在实现中被零除

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Custom dataloaders
    train_dataset = AiShellDataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               pin_memory=True, shuffle=True, num_workers=num_workers)
    valid_dataset = AiShellDataset(args, 'dev')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               pin_memory=True, shuffle=False,
                                               num_workers=num_workers)  # pin_memory=True：生成的Tensor数据最开始是属于内存中  collate_fn=pad_collate：将一个list的sample组成一个mini-batch的函数

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)
        writer.add_scalar('model/train_loss', train_loss, epoch)

        lr = optimizer.lr
        print('\nLearning rate: {}'.format(lr))
        step_num = optimizer.step_num
        print('Step num: {}\n'.format(step_num))

        writer.add_scalar('model/learning_rate', lr, epoch)

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           logger=logger)
        writer.add_scalar('model/valid_loss', valid_loss, epoch)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        if epoch%1 == 0 and epoch != 0:
            save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)
            np.save("train_losses.npy", train_losses)
            np.save("valid_losses.npy", valid_losses)


def train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        # Forward prop.
        loss, att_w = model(padded_input, input_lengths.to('cpu'), padded_target.long())

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        #plot the alignment figure
        if epoch%1 ==0 and epoch!=0 and i == len(train_loader)-1:
            save_and_trace_attention1(att_w)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

    return losses.avg


def valid(valid_loader, model, logger):
    model.eval()

    losses = AverageMeter()

    # Batches
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        # Forward prop.
        loss, att_w = model(padded_input, input_lengths.to('cpu'), padded_target.long())

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    logger.info('\nValidation Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses))

    return losses.avg


def main():
    global args
    args = parser.parse_args()
    train_net(args)



if __name__ == '__main__':
    data_pre()
    main()
