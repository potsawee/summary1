import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
import pickle
import random
from datetime import datetime

from data_meeting import TopicSegment, Utterance, bert_tokenizer
from models.hierarchical_rnn import EncoderDecoder

def train1():
    print("Start training hierarchical RNN model")
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['use_gpu'] = True
    args['num_utterances'] = 2000  # max no. utterance in a meeting
    args['num_words']      = 64    # max no. words in an utterance
    args['summary_length'] = 800   # max no. words in a summary
    args['vocab_size']     = 30522 # BERT tokenizer
    args['embedding_dim']  = 256   # word embeeding dimension
    args['rnn_hidden_size'] = 300 # RNN hidden size

    args['batch_size']      = 1
    args['update_nbatches'] = 2
    args['num_epochs']      = 1000
    args['random_seed']     = 28
    args['best_val_loss']     = 1e+10
    args['val_batch_size']    = 4
    args['val_stop_training'] = 10

    args['lr']        = 0.1
    args['adjust_lr'] = True # if True overwrite the learning rate above
    args['initial_lr'] = 0.1 # lr = lr_0*step^(-decay_rate)
    args['decay_rate'] = 0.5

    args['model_save_dir'] = "/home/alta/summary/pm574/summariser1/lib/trained_models/"
    args['model_name'] = 'HGRU_DEC8A'
    # ---------------------------------------------------------------------------------- #

    if args['use_gpu']:
        if 'X_SGE_CUDA_DEVICE' in os.environ: # to run on CUED stack machine
            print('running on the stack...')
            cuda_device = os.environ['X_SGE_CUDA_DEVICE']
            print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
        else:
            # pdb.set_trace()
            print('running locally...')
            os.environ["CUDA_VISIBLE_DEVICES"] = '1' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    print("device = {}".format(device))

    # random seed
    random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    np.random.seed(args['random_seed'])

    train_data = load_ami_data('train')
    valid_data = load_ami_data('valid')

    model = EncoderDecoder(args, device=device)
    print(model)

    # Hyperparameters
    BATCH_SIZE = args['batch_size']
    NUM_EPOCHS = args['num_epochs']
    VAL_BATCH_SIZE = args['val_batch_size']
    VAL_STOP_TRAINING = args['val_stop_training']

    criterion = nn.NLLLoss(reduction='none')

    # we use two separate optimisers (encoder & decoder)
    optimizer = optim.Adam(model.parameters(),lr=args['lr'],betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer.zero_grad()

    # validation losses
    best_val_loss = args['best_val_loss']
    best_epoch    = 0
    stop_counter  = 0

    for epoch in range(NUM_EPOCHS):
        print("======================= Training epoch {} =======================".format(epoch))
        num_train_data = len(train_data)
        # num_batches = int(num_train_data/BATCH_SIZE) + 1
        num_batches = int(num_train_data/BATCH_SIZE)
        print("num_batches = {}".format(num_batches))

        print("shuffle train data")
        random.shuffle(train_data)

        idx = 0

        for bn in range(num_batches):

            input, u_len, w_len, target, tgt_len = get_a_batch(
                    train_data, idx, BATCH_SIZE,
                    args['num_utterances'], args['num_words'],
                    args['summary_length'], device)

            # decoder target
            decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device)
            decoder_target = decoder_target.view(-1)
            decoder_mask = decoder_mask.view(-1)

            decoder_output = model(input, u_len, w_len, target)

            loss = criterion(decoder_output.view(-1, args['vocab_size']), decoder_target)
            loss = (loss * decoder_mask).sum() / decoder_mask.sum()
            loss.backward()

            idx += BATCH_SIZE

            if bn % args['update_nbatches'] == 0:
                # gradient_clipping
                max_norm = 0.5
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                # update the gradients
                if args['adjust_lr']:
                    adjust_lr(optimizer, epoch, num_batches, bn, args['initial_lr'], args['decay_rate'])
                optimizer.step()
                optimizer.zero_grad()

            if bn % 1 == 0:
                print("[{}] batch number {}/{}: loss = {}".format(str(datetime.now()), bn, num_batches, loss))
                sys.stdout.flush()

            if bn % 10 == 0:

                print("======================== GENERATED SUMMARY ========================")
                print(bert_tokenizer.decode(torch.argmax(decoder_output[0], dim=-1).cpu().numpy()[:tgt_len[0]]))
                print("======================== REFERENCE SUMMARY ========================")
                print(bert_tokenizer.decode(decoder_target.view(BATCH_SIZE,args['summary_length'])[0,:tgt_len[0]].cpu().numpy()))

            if bn == 0: # e.g. eval every epoch
                # ---------------- Evaluate the model on validation data ---------------- #
                print("Evaluating the model at epoch {} step {}".format(epoch, bn))
                print("learning_rate = {}".format(optimizer.param_groups[0]['lr']))
                model.eval() # switch to evaluation mode
                with torch.no_grad():
                    avg_val_loss = evaluate(model, valid_data, VAL_BATCH_SIZE, args, device)
                print("avg_val_loss_per_token = {}".format(avg_val_loss))
                model.train() # switch to training mode
                # ------------------- Save the model OR Stop training ------------------- #
                if avg_val_loss < best_val_loss:
                    stop_counter = 0
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    savepath = args['model_save_dir']+"model-{}-ep{}.pt".format(args['model_name'],epoch)
                    torch.save(model.state_dict(), savepath)
                    print("Model improved & saved at {}".format(savepath))
                else:
                    print("Model not improved #{}".format(stop_counter))
                    if stop_counter < VAL_STOP_TRAINING:
                        # load the previous model
                        latest_model = args['model_save_dir']+"model-{}-ep{}.pt".format(args['model_name'],best_epoch)
                        model.load_state_dict(torch.load(latest_model))
                        model.train()
                        print("Restored model from {}".format(latest_model))
                        stop_counter += 1

                    else:
                        print("Model has not improved for {} times! Stop training.".format(VAL_STOP_TRAINING))
    print("End of training hierarchical RNN model")

def evaluate(model, eval_data, eval_batch_size, args, device):
    # num_eval_epochs = int(eval_data['num_data']/eval_batch_size) + 1
    num_eval_epochs = int(len(eval_data)/eval_batch_size)

    print("num_eval_epochs = {}".format(num_eval_epochs))

    eval_idx = 0
    eval_total_loss = 0.0
    eval_total_tokens = 0

    criterion = nn.NLLLoss(reduction='none')

    for bn in range(num_eval_epochs):

        input, u_len, w_len, target, tgt_len = get_a_batch(
                eval_data, eval_idx, eval_batch_size,
                args['num_utterances'], args['num_words'],
                args['summary_length'], device)

        # decoder target
        decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device)
        decoder_target = decoder_target.view(-1)
        decoder_mask = decoder_mask.view(-1)

        decoder_output = model(input, u_len, w_len, target)

        loss = criterion(decoder_output.view(-1, args['vocab_size']), decoder_target)
        eval_total_loss += (loss * decoder_mask).sum().item()
        eval_total_tokens += decoder_mask.sum().item()

        eval_idx += eval_batch_size

        print("#", end="")
        sys.stdout.flush()

    print()
    avg_eval_loss = eval_total_loss / eval_total_tokens
    return avg_eval_loss


def adjust_lr(optimizer, epoch, epoch_size, bn, lr0, decay_rate):
    """to adjust the learning rate for both encoder & decoder"""
    step = (epoch * epoch_size) + bn + 1 # plus 1 to avoid ZeroDivisionError
    # lr = min(1e-3, 0.05*step**(-1.25))
    lr = lr0*step**(-decay_rate)

    for param_group in optimizer.param_groups: param_group['lr'] = lr
    return

def shift_decoder_target(target, tgt_len, device):
    # MASK_TOKEN_ID = 103
    batch_size = target.size(0)
    max_len = target.size(1)
    dtype0  = target.dtype

    decoder_target = torch.zeros((batch_size, max_len), dtype=dtype0, device=device)
    decoder_target[:,:-1] = target.clone().detach()[:,1:]
    # decoder_target[:,-1:] = 103 # MASK_TOKEN_ID = 103
    # decoder_target[:,-1:] = 0 # add padding id instead of MASK

    # mask for shifted decoder target
    decoder_mask = torch.zeros((batch_size, max_len), dtype=torch.float, device=device)
    for bn, l in enumerate(tgt_len):
        decoder_mask[bn,:l-1].fill_(1.0)

    return decoder_target, decoder_mask

def get_a_batch(ami_data, idx, batch_size, num_utterances, num_words, summary_length, device):
    input   = torch.zeros((batch_size, num_utterances, num_words), dtype=torch.long)
    summary = torch.zeros((batch_size, summary_length), dtype=torch.long)

    # utt_lengths  = torch.zeros((batch_size), dtype=torch.int)
    # word_lengths = torch.zeros((batch_size, num_utterances), dtype=torch.int)
    utt_lengths  = np.zeros((batch_size), dtype=np.int)
    word_lengths = np.zeros((batch_size, num_utterances), dtype=np.int)

    # summary lengths
    summary_lengths = np.zeros((batch_size), dtype=np.int)


    for bn in range(batch_size):
        topic_segments  = ami_data[idx+bn][0]
        encoded_summary = ami_data[idx+bn][1]
        # input
        utt_id = 0
        for segment in topic_segments:
            utterances = segment.utterances
            for utterance in utterances:
                encoded_words = utterance.encoded_words
                l = len(encoded_words)
                if l > num_words:
                    encoded_words = encoded_words[:num_words]
                    l = num_words
                input[bn,utt_id,:l] = torch.tensor(encoded_words)
                # word_lengths[bn,utt_id] = torch.tensor(l)
                word_lengths[bn,utt_id] = l
                utt_id += 1

                if utt_id == num_utterances: break
            if utt_id == num_utterances: break

        # utt_lengths[bn] = torch.tensor(utt_id)
        utt_lengths[bn] = utt_id

        # summary
        l = len(encoded_summary)
        summary_lengths[bn] = l
        summary[bn, :l] = torch.tensor(encoded_summary)

    input   = input.to(device)
    summary = summary.to(device)
    # utt_lengths  = utt_lengths.to(device)
    # word_lengths = word_lengths.to(device)

    return input, utt_lengths, word_lengths, summary, summary_lengths

def load_ami_data(data_type):
    path = "lib/model_data/ami-191206.{}.pk.bin".format(data_type)
    with open(path, 'rb') as f:
        ami_data = pickle.load(f, encoding="bytes")
    return ami_data


if __name__ == "__main__":
    # ------ TRAINING ------ #
    train1()
