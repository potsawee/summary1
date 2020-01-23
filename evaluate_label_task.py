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
from collections import OrderedDict

from data_meeting import TopicSegment, Utterance, bert_tokenizer, DA_MAPPING
from data import cnndm
from data.cnndm import ProcessedDocument, ProcessedSummary
from models.hierarchical_rnn_v2 import EncoderDecoder, DALabeller, EXTLabeller
from models.neural import LabelSmoothingLoss
from train1 import print_config, load_ami_data, get_a_batch, shift_decoder_target, length2mask

def evaluate_label_task(model_name, epoch):
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['use_gpu']        = False
    args['air_multi_gpu']  = False  # to enable running on multiple GPUs on stack
    args['num_utterances'] = 2000  # max no. utterance in a meeting
    args['num_words']      = 64    # max no. words in an utterance
    args['summary_length'] = 800   # max no. words in a summary
    args['summary_type']   = 'long'   # long or short summary
    args['vocab_size']     = 30522 # BERT tokenizer
    args['embedding_dim']   = 256   # word embeeding dimension
    args['rnn_hidden_size'] = 512 # RNN hidden size

    args['dropout']        = 0.5
    args['num_layers_enc'] = 1    # in total it's num_layers_enc*3 (word/utt/seg)
    args['num_layers_dec'] = 1

    args['batch_size']      = 2
    args['update_nbatches'] = 2   # 0 meaning whole batch update & using SGD

    args['model_save_dir'] = "/home/alta/summary/pm574/summariser1/lib/trained_models/"
    args['load_model'] = "/home/alta/summary/pm574/summariser1/lib/trained_models/model-{}-ep{}".format(model_name, epoch)
    # ---------------------------------------------------------------------------------- #

    if args['use_gpu']:
        if 'X_SGE_CUDA_DEVICE' in os.environ: # to run on CUED stack machine
            if not args['air_multi_gpu']:
                print('running on the stack... 1 GPU')
                cuda_device = os.environ['X_SGE_CUDA_DEVICE']
                print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
            else:
                print('running on the stack... multiple GPUs')
                os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
                write_multi_sl(args['model_name'])
        else:
            print('running locally...')
            os.environ["CUDA_VISIBLE_DEVICES"] = '3' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'

    train_data = load_ami_data('test')
    valid_data = load_ami_data('valid')

    model = EncoderDecoder(args, device=device)
    NUM_DA_TYPES = len(DA_MAPPING)
    da_labeller = DALabeller(args['rnn_hidden_size'], NUM_DA_TYPES, device)
    ext_labeller = EXTLabeller(args['rnn_hidden_size'], device)

    # to use multiple GPUs
    if device == 'cuda':
        try:
            model.load_state_dict( torch.load(args['load_model']+'.pt') )
            da_labeller.load_state_dict( torch.load(args['load_model']+'.da.pt') )
            ext_labeller.load_state_dict(torch.load(args['load_model']+'.ext.pt'))
        except:
            model_state_dict = torch.load( args['load_model']+'.pt' )
            new_model_state_dict = OrderedDict()
            for key in model_state_dict.keys():
                new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
            model.load_state_dict(new_model_state_dict)

            model_state_dict = torch.load(args['load_model']+'.da.pt')
            new_model_state_dict = OrderedDict()
            for key in model_state_dict.keys():
                new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
            da_labeller.load_state_dict(new_model_state_dict)

            model_state_dict = torch.load(args['load_model']+'.ext.pt')
            new_model_state_dict = OrderedDict()
            for key in model_state_dict.keys():
                new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
            ext_labeller.load_state_dict(new_model_state_dict)
    else:
        try:
            model.load_state_dict(torch.load(args['load_model']+'.pt',  map_location=torch.device('cpu')))
            da_labeller.load_state_dict(torch.load(args['load_model']+'.da.pt', map_location=torch.device('cpu')))
            ext_labeller.load_state_dict(torch.load(args['load_model']+'.ext.pt', map_location=torch.device('cpu')))
        except:
            model_state_dict = torch.load(args['load_model']+'.pt', map_location=torch.device('cpu'))
            new_model_state_dict = OrderedDict()
            for key in model_state_dict.keys():
                new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
            model.load_state_dict(new_model_state_dict)

            model_state_dict = torch.load(args['load_model']+'.da.pt', map_location=torch.device('cpu'))
            new_model_state_dict = OrderedDict()
            for key in model_state_dict.keys():
                new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
            da_labeller.load_state_dict(new_model_state_dict)

            model_state_dict = torch.load(args['load_model']+'.ext.pt', map_location=torch.device('cpu'))
            new_model_state_dict = OrderedDict()
            for key in model_state_dict.keys():
                new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
            ext_labeller.load_state_dict(new_model_state_dict)

    BATCH_SIZE = args['batch_size']
    num_train_data = len(train_data)

    num_batches = int(num_train_data/BATCH_SIZE)

    idx = 0

    model = model.eval()
    da_labeller = da_labeller.eval()
    ext_labeller = ext_labeller.eval()

    ts_tp = 0
    ts_tn = 0
    ts_fp = 0
    ts_fn = 0

    da_true = 0
    da_total = 0

    ext_tp = 0
    ext_tn = 0
    ext_fp = 0
    ext_fn = 0

    for bn in range(num_batches):

        input, u_len, w_len, target, tgt_len, topic_boundary_label, dialogue_acts, extractive_label = get_a_batch(
                train_data, idx, BATCH_SIZE,
                args['num_utterances'], args['num_words'],
                args['summary_length'], args['summary_type'], device)

        # decoder target
        decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device, mask_offset=True)
        decoder_target = decoder_target.view(-1)
        decoder_mask = decoder_mask.view(-1)

        decoder_output, gate_z, u_output = model(input, u_len, w_len, target)

        # multitask(1): topic segmentation prediction
        loss_ts_mask = length2mask(u_len, BATCH_SIZE, args['num_utterances'], device)

        # multitask(2): dialogue act prediction
        da_output = da_labeller(u_output)

        # multitask(3): extractive label prediction
        ext_output = ext_labeller(u_output).squeeze(-1)

        tp, tn, fp ,fn = labelling_eval(gate_z, topic_boundary_label, loss_ts_mask)
        ts_tp += tp
        ts_tn += tn
        ts_fp += fp
        ts_fn += fn

        t, total = multiclass_eval(torch.argmax(da_output,dim=-1), dialogue_acts, loss_ts_mask)
        da_true += t
        da_total += total

        tp, tn, fp ,fn = labelling_eval(ext_output, extractive_label, loss_ts_mask)
        ext_tp += tp
        ext_tn += tn
        ext_fp += fp
        ext_fn += fn
        idx += BATCH_SIZE
        print("#",end='')
        sys.stdout.flush()

    print()

    print("Model:", args['load_model'])
    print("[1] ========== Topic Segmentation Task ==========")
    try:
        accuracy = (ts_tp+ts_tn)/(ts_tp+ts_tn+ts_fp+ts_fn)
        precision = ts_tp/(ts_tp+ts_fp)
        recall = ts_tp/(ts_tp+ts_fn)
        f1     = precision*recall / (precision+recall)
        print("Acc: {:.4f} | Pre: {:.4f} | Rec: {:.4f} | F-1: {:.4f}".format(accuracy,precision,recall,f1))
    except ZeroDivisionError:
        print("zerodivision")

    print("[2] ======= Dialogue Act Prediction Task =======")
    try:
        accuracy = da_true / da_total
        print("Acc: {:.4f}".format(accuracy))
    except ZeroDivisionError:
        print("zerodivision")
    print("[3] ============== Extractive Task ==============")
    try:
        accuracy = (ext_tp+ext_tn)/(ext_tp+ext_tn+ext_fp+ext_fn)
        precision = ext_tp/(ext_tp+ext_fp)
        recall = ext_tp/(ext_tp+ext_fn)
        f1     = precision*recall / (precision+recall)
        print("Acc: {:.4f} | Pre: {:.4f} | Rec: {:.4f} | F-1: {:.4f}".format(accuracy,precision,recall,f1))
    except ZeroDivisionError:
        print("zerodivision")
def multiclass_eval(output, target, mask):
    # evaluate accuracy
    match_arr = (output == target).type(torch.FloatTensor)
    match = (match_arr * mask).sum().item()
    total = mask.sum().item()
    return match, total


def labelling_eval(output, label, mask):
    # evaluate P, R, F1-score, accuracy
    output = output.view(-1)
    label = label.view(-1)
    mask = mask.view(-1)
    size = output.size(0)
    pred = torch.zeros((size), dtype=torch.float)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(size):
        if mask[i] == 0.0: continue
        if label[i] == 1.0: # positive
            if output[i] > 0.5: tp += 1
            else: fn += 1
        else: # negative
            if output[i] > 0.5: fp += 1
            else: tn += 1
    return tp, tn, fp ,fn

def main():
    model_name = "HGRUV2_CNNDM_AMI_JAN23A"
    epochs = [16]
    # epochs = [21]
    for ep in epochs:
        try:
            evaluate_label_task(model_name, ep)
        except:
            print("ERROR: model-{}-ep{}".format(model_name, ep))
main()
