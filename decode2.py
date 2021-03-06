"""Inference time script for the abstractive task"""
import os
import sys
import torch
import pdb
import numpy as np
import random
from datetime import datetime

from models.pgnet import PointerGeneratorNetwork
from train1 import get_a_batch, load_ami_data
from data_meeting import TopicSegment, Utterance, bert_tokenizer

if torch.__version__ == '1.2.0': KEYPADMASK_DTYPE = torch.bool
else:
    print("source ~/anaconda3/bin/activate torch12-cuda10")
    raise Exception("Torch Version not supported")

START_TOKEN = '[CLS]'
SEP_TOKEN   = '[SEP]'
STOP_TOKEN  = '[MASK]'

START_TOKEN_ID = bert_tokenizer.convert_tokens_to_ids(START_TOKEN)
SEP_TOKEN_ID   = bert_tokenizer.convert_tokens_to_ids(SEP_TOKEN)
STOP_TOKEN_ID  = bert_tokenizer.convert_tokens_to_ids(STOP_TOKEN)

TEST_DATA_SIZE = 20
VOCAB_SIZE     = 30522

def decoding(model, data, args, start_idx, batch_size, num_batches, k, search_method, alpha):
    device = args['device']
    max_summary_length = args['summary_length']
    time_step = max_summary_length
    idx = 0
    summary_out_dir = args['summary_out_dir']

    alpha = alpha
    length_offset = 5

    decode_dict = {
        'k': k, 'search_method': search_method,
        'time_step': time_step, 'vocab_size': VOCAB_SIZE,
        'device': device, 'start_token_id': START_TOKEN_ID,
        'stop_token_id': STOP_TOKEN_ID,
        'alpha': alpha, 'length_offset': length_offset,
        'keypadmask_dtype': KEYPADMASK_DTYPE
    }

    for bn in range(num_batches):
        decode_dict['batch_size'] = batch_size

        input, u_len, w_len, target, tgt_len  = get_a_batch(
            data, start_idx+idx, batch_size, args['num_utterances'],
            args['num_words'], args['summary_length'], args['summary_type'], device)

        if args['decode_method'] == 'beamsearch':
            summaries_id = model.decode_beamsearch(input, u_len, w_len, decode_dict)
        if args['decode_method'] == 'sampling':
            summaries_id = model.decode_sampling(input, u_len, w_len, decode_dict)

        # finish t = 0,...,max_summary_length
        summaries = [None for _ in range(batch_size)]
        for j in range(batch_size):
            summaries[j] = tgtids2summary(summaries_id[j])

        write_summary_files(summary_out_dir, summaries, start_idx+idx)

        print("[{}] batch {}/{} --- idx [{},{})".format(
                str(datetime.now()), bn+1, num_batches,
                start_idx+idx, start_idx+idx+batch_size))

        sys.stdout.flush()
        idx += batch_size


def write_summary_files(dir, summaries, start_idx):
    if not os.path.exists(dir): os.makedirs(dir)
    num_data = len(summaries)
    for idx in range(num_data):
        filepath = dir + 'file.{}.txt'.format(idx+start_idx)
        line = '\n'.join(summaries[idx])
        with open(filepath, 'w') as f:
            f.write(line)

def tgtids2summary(tgt_ids):
    # tgt_ids = a row of numpy array containing token ids
    bert_decoded = bert_tokenizer.decode(tgt_ids)
    # truncate START_TOKEN & part after STOP_TOKEN
    stop_idx = bert_decoded.find(STOP_TOKEN)
    processed_bert_decoded = bert_decoded[5:stop_idx]
    summary = [s.strip() for s in processed_bert_decoded.split(SEP_TOKEN)]
    return summary

def decode(start_idx):
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['use_gpu']        = True
    args['num_utterances'] = 2000  # max no. utterance in a meeting
    args['num_words']      = 64    # max no. words in an utterance
    args['summary_length'] = 250   # max no. words in a summary
    args['summary_type']   = 'short'   # max no. words in a summary
    args['vocab_size']     = 30522 # BERT tokenizer
    args['embedding_dim']   = 128   # word embeeding dimension
    args['rnn_hidden_size'] = 256 # RNN hidden size
    args['num_words_meeting'] = 8400

    args['dropout']        = 0.0
    args['num_layers_enc'] = 1
    args['num_layers_dec'] = 1

    args['model_save_dir'] = "/home/alta/summary/pm574/summariser1/lib/trained_models/"
    args['model_data_dir'] = "/home/alta/summary/pm574/summariser1/lib/model_data/"

    args['model_name'] = "PGN_DEC16F"
    args['model_epoch'] = 10
    # ---------------------------------------------------------------------------------- #
    start_idx   = start_idx
    batch_size  = 1
    num_batches = 5
    args['decode_method'] = 'beamsearch'
    search_method = 'argmax'
    beam_width  = 5
    alpha       = 1.25
    random_seed = 28
    # ---------------------------------------------------------------------------------- #
    if args['decode_method'] == 'sampling':
        args['summary_out_dir'] = \
        '/home/alta/summary/pm574/summariser1/out_summary/model-{}-ep{}/sampling/' \
            .format(args['model_name'], args['model_epoch'])

    elif args['decode_method'] == 'beamsearch':
        args['summary_out_dir'] = \
        '/home/alta/summary/pm574/summariser1/out_summary/model-{}-ep{}/width{}-{}-alpha{}/' \
            .format(args['model_name'], args['model_epoch'], beam_width, search_method, alpha)
    # ---------------------------------------------------------------------------------- #
    if args['use_gpu']:
        if 'X_SGE_CUDA_DEVICE' in os.environ: # to run on CUED stack machine
            print('running on the stack...')
            cuda_device = os.environ['X_SGE_CUDA_DEVICE']
            print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
        else:
            print('running locally...')
            os.environ["CUDA_VISIBLE_DEVICES"] = '0' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    args['device'] = device
    print("device = {}".format(device))

    # random seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Define and Load the model
    model = PointerGeneratorNetwork(args, device)
    trained_model = args['model_save_dir']+"model-{}-ep{}.pt".format(args['model_name'],args['model_epoch'])
    if device == 'cuda':
        model.load_state_dict(torch.load(trained_model))
    elif device == 'cpu':
        model.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu')))

    model.eval() # switch it to eval mode
    print("Restored model from {}".format(trained_model))

    # Load and prepare data
    test_data = load_ami_data('test')
    print("========================================================")
    print("start decoding: idx [{},{})".format(start_idx, start_idx + batch_size*num_batches))
    print("========================================================")

    with torch.no_grad():
        print("beam_width = {}".format(beam_width))
        decoding(model, test_data, args, start_idx, batch_size, num_batches,
                 k=beam_width, search_method=search_method, alpha=alpha)

    print("finish decoding: idx [{},{})".format(start_idx, start_idx + batch_size*num_batches))


if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("Usage: python decode.py start_idx")
        raise Exception("argv error")

    start_idx = int(sys.argv[1])
    decode(start_idx)
