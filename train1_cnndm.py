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

from data_meeting import TopicSegment, Utterance, bert_tokenizer, DA_MAPPING
from data import cnndm
from data.cnndm import ProcessedDocument, ProcessedSummary
from models.hierarchical_rnn_v2 import EncoderDecoder, DALabeller, EXTLabeller
from models.neural import LabelSmoothingLoss

def train1():
    print("Start training hierarchical RNN model")
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['use_gpu']        = True
    args['air_multi_gpu']  = False  # to enable running on multiple GPUs on stack
    args['num_utterances'] = 50  # max no. utterance in a meeting
    args['num_words']      = 32    # max no. words in an utterance
    args['summary_length'] = 144   # max no. words in a summary
    args['summary_type']   = 'long'   # long or short summary
    args['vocab_size']     = 30522 # BERT tokenizer
    args['embedding_dim']   = 256   # word embeeding dimension
    args['rnn_hidden_size'] = 512 # RNN hidden size

    args['dropout']        = 0.5
    args['num_layers_enc'] = 1    # in total it's num_layers_enc*3 (word/utt/seg)
    args['num_layers_dec'] = 1

    args['batch_size']      = 64
    args['update_nbatches'] = 1   # 0 meaning whole batch update & using SGD
    args['num_epochs']      = 20
    args['random_seed']     = 78
    args['best_val_loss']     = 1e+10
    args['val_batch_size']    = 64
    args['val_stop_training'] = 10

    args['lr']         = 0.01
    args['adjust_lr']  = True       # if True overwrite the learning rate above
    args['initial_lr'] = 1e-2       # lr = lr_0*step^(-decay_rate)
    args['decay_rate'] = 0.25
    args['label_smoothing'] = 0.1

    args['model_save_dir'] = "/home/alta/summary/pm574/summariser1/lib/trained_models/"
    # args['load_model'] = "/home/alta/summary/pm574/summariser1/lib/trained_models/model-HGRUV2_CNNDM_JAN10A-ep0-bn5000.pt"
    args['load_model'] = None
    args['model_name'] = 'HGRUV2_CNNDM_JAN26A'
    # ---------------------------------------------------------------------------------- #
    print_config(args)


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
            os.environ["CUDA_VISIBLE_DEVICES"] = '0,1' # choose the device (GPU) here
        device = 'cuda'
    else:
        device = 'cpu'
    print("device = {}".format(device))


    # random seed
    random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    np.random.seed(args['random_seed'])

    dataset = 'cnndm'

    if dataset == 'cnndm':
        import pickle
        args['model_data_dir'] = "/home/alta/summary/pm574/summariser0/lib/model_data/"
        args['max_pos_embed']      = 512
        args['max_num_sentences']  = 32
        args['max_summary_length'] = args['summary_length']
        train_data = load_cnndm_data(args, 'trainx', dump=False)
        valid_data = load_cnndm_data(args, 'valid',  dump=False)
    else:
        raise Exception("CNNDM pretraining!")

    print("{} loaded".format(dataset))

    model = EncoderDecoder(args, device=device)
    print(model)

    # to use multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Multiple GPUs: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    # Load model if specified (path to pytorch .pt)
    if args['load_model'] != None:
        model.load_state_dict(torch.load(args['load_model']))
        model.train()
        print("Loaded model from {}".format(args['load_model']))
    else:
        print("Train a new model")


    # Hyperparameters
    BATCH_SIZE = args['batch_size']
    NUM_EPOCHS = args['num_epochs']
    VAL_BATCH_SIZE = args['val_batch_size']
    VAL_STOP_TRAINING = args['val_stop_training']

    if args['label_smoothing'] > 0.0:
        criterion = LabelSmoothingLoss(num_classes=args['vocab_size'],
                        smoothing=args['label_smoothing'], reduction='none')
    else:
        criterion = nn.NLLLoss(reduction='none')

    # we use two separate optimisers (encoder & decoder)
    optimizer = optim.Adam(model.parameters(),lr=args['lr'],betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer.zero_grad()
    sgd_optimizer = optim.SGD(model.parameters(), lr=args['lr'])
    sgd_optimizer.zero_grad()

    # validation losses
    best_val_loss = args['best_val_loss']
    best_epoch    = 0
    best_bn       = 0
    stop_counter  = 0

    training_step = 0

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
                    args['summary_length'], args['summary_type'], device)

            # decoder target
            decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device, mask_offset=True)
            decoder_target = decoder_target.view(-1)
            decoder_mask = decoder_mask.view(-1)

            try:
                decoder_output, _, _ = model(input, u_len, w_len, target)
            except IndexError:
                print("there is an IndexError --- likely from if segment_indices[bn][-1] == u_len[bn]-1:")
                print("for now just skip this batch!")
                idx += BATCH_SIZE # previously I forget to add this line!!!
                continue

            loss = criterion(decoder_output.view(-1, args['vocab_size']), decoder_target)
            loss = (loss * decoder_mask).sum() / decoder_mask.sum()

            loss.backward()

            idx += BATCH_SIZE

            if args['update_nbatches'] != 0:
                if bn % args['update_nbatches'] == 0:
                    # gradient_clipping
                    max_norm = 0.5
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    # update the gradients
                    if args['adjust_lr']:
                        adjust_lr(optimizer, args['initial_lr'], args['decay_rate'], training_step)
                    optimizer.step()
                    optimizer.zero_grad()
                    training_step += 1
            else:
                # whole data set update
                if bn == num_batches-1:
                    # update the gradients
                    sgd_optimizer.step()
                    sgd_optimizer.zero_grad()

            if bn % 5 == 0:
                print("[{}] batch {}/{}: loss = {:5f}".
                    format(str(datetime.now()), bn, num_batches, loss))
                sys.stdout.flush()

            if bn % 50 == 0:

                print("======================== GENERATED SUMMARY ========================")
                print(bert_tokenizer.decode(torch.argmax(decoder_output[0], dim=-1).cpu().numpy()[:tgt_len[0]]))
                print("======================== REFERENCE SUMMARY ========================")
                print(bert_tokenizer.decode(decoder_target.view(BATCH_SIZE,args['summary_length'])[0,:tgt_len[0]].cpu().numpy()))

            if bn % 500 == 0:
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
                    best_bn = bn
                    savepath = args['model_save_dir']+"model-{}-ep{}-bn{}.pt".format(args['model_name'],epoch,bn)
                    torch.save(model.state_dict(), savepath)
                    print("Model improved & saved at {}".format(savepath))
                else:
                    print("Model not improved #{}".format(stop_counter))
                    if stop_counter < VAL_STOP_TRAINING:
                        # load the previous model
                        latest_model = args['model_save_dir']+"model-{}-ep{}-bn{}.pt".format(args['model_name'],best_epoch,best_bn)
                        model.load_state_dict(torch.load(latest_model))
                        model.train()
                        print("Restored model from {}".format(latest_model))
                        stop_counter += 1

                    else:
                        print("Model has not improved for {} times! Stop training.".format(VAL_STOP_TRAINING))
                        return

    if args['air_multi_gpu']: rm_multi_sl(args['model_name'])
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
                args['summary_length'], args['summary_type'], device)

        # decoder target
        decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device)
        decoder_target = decoder_target.view(-1)
        decoder_mask = decoder_mask.view(-1)

        decoder_output, _, _ = model(input, u_len, w_len, target)

        loss = criterion(decoder_output.view(-1, args['vocab_size']), decoder_target)
        eval_total_loss += (loss * decoder_mask).sum().item()
        eval_total_tokens += decoder_mask.sum().item()

        eval_idx += eval_batch_size

        print("#", end="")
        sys.stdout.flush()

        # if bn == 20: break

    print()
    avg_eval_loss = eval_total_loss / eval_total_tokens
    return avg_eval_loss

def adjust_lr(optimizer, lr0, decay_rate, step):
    """to adjust the learning rate for both encoder & decoder --- DECAY"""
    step = step + 1 # plus 1 to avoid ZeroDivisionError
    # lr = min(1e-3, 0.05*step**(-1.25))
    lr = lr0*step**(-decay_rate)

    for param_group in optimizer.param_groups: param_group['lr'] = lr
    return

def shift_decoder_target(target, tgt_len, device, mask_offset=False):
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
    if mask_offset:
        offset = 10
        for bn, l in enumerate(tgt_len):
            # decoder_mask[bn,:l-1].fill_(1.0)
            # to accommodate like 10 more [MASK] [MASK] [MASK] [MASK],...
            if l-1+offset < max_len: decoder_mask[bn,:l-1+offset].fill_(1.0)
            else: decoder_mask[bn,:].fill_(1.0)
    else:
        for bn, l in enumerate(tgt_len):
            decoder_mask[bn,:l-1].fill_(1.0)

    return decoder_target, decoder_mask

def get_a_batch(ami_data, idx, batch_size, num_utterances, num_words, summary_length, sum_type, device):
    if sum_type not in ['long', 'short']:
        raise Exception("summary type long/short only")

    input   = torch.zeros((batch_size, num_utterances, num_words), dtype=torch.long)
    summary = torch.zeros((batch_size, summary_length), dtype=torch.long)
    summary.fill_(103)

    utt_lengths  = np.zeros((batch_size), dtype=np.int)
    word_lengths = np.zeros((batch_size, num_utterances), dtype=np.int)

    # summary lengths
    summary_lengths = np.zeros((batch_size), dtype=np.int)

    for bn in range(batch_size):
        topic_segments  = ami_data[idx+bn][0]
        if sum_type == 'long':
            encoded_summary = ami_data[idx+bn][1]
        elif sum_type == 'short':
            encoded_summary = ami_data[idx+bn][2]
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
        if l > summary_length:
            encoded_summary = encoded_summary[:summary_length]
            l = summary_length
        summary_lengths[bn] = l
        summary[bn, :l] = torch.tensor(encoded_summary)

    input   = input.to(device)
    summary = summary.to(device)

    # covert numpy to torch tensor (for multiple GPUs purpose)
    utt_lengths = torch.from_numpy(utt_lengths)
    word_lengths = torch.from_numpy(word_lengths)
    summary_lengths = torch.from_numpy(summary_lengths)

    return input, utt_lengths, word_lengths, summary, summary_lengths

def load_ami_data(data_type):
    path = "/home/alta/summary/pm574/summariser1/lib/model_data/ami-191209.{}.pk.bin".format(data_type)
    with open(path, 'rb') as f:
        ami_data = pickle.load(f, encoding="bytes")
    return ami_data

def load_cnndm_data(args, data_type, dump=False):
    if dump:
        data    = cnndm.load_data(args, data_type)
        summary = cnndm.load_summary(args, data_type)
        articles = []
        for encoded_words in data['encoded_articles']:
            # encoded_sentences = []
            article = TopicSegment()
            l = len(encoded_words) - 1
            for i, x in enumerate(encoded_words):
                if x == 101: # CLS
                    sentence = []
                elif x == 102: # SEP
                    utt = Utterance(sentence, -1, -1, -1)
                    article.add_utterance(utt)
                elif x == 100: # UNK
                    break
                else:
                    sentence.append(x)
                    if i == l:
                        utt = Utterance(sentence, -1, -1, -1)
                        article.add_utterance(utt)
            articles.append([article])
        abstracts = []
        for encoded_abstract in summary['encoded_abstracts']:
            if 103 in encoded_abstract:
                last_idx = encoded_abstract.index(103)
                encoded_abstract = encoded_abstract[:last_idx]
            encoded_abstract.append(102)
            encoded_abstract.append(103)
            abstracts.append(encoded_abstract)
        cnndm_data = []
        for x, y in zip(articles, abstracts):
            cnndm_data.append((x,y,y))
    else:
        path = "/home/alta/summary/pm574/summariser1/lib/model_data/cnndm-191216.{}.pk.bin".format(data_type)
        with open(path, 'rb') as f:
            cnndm_data = pickle.load(f, encoding="bytes")

    return cnndm_data

def print_config(args):
    print("============================= CONFIGURATION =============================")
    for x in args:
        print('{}={}'.format(x, args[x]))
    print("=========================================================================")

def write_multi_sl(fname):
    path = "/home/alta/summary/pm574/temp/{}".format(fname)
    with open(path, 'w') as f: f.write(fname)
def rm_multi_sl(fname):
    path = "/home/alta/summary/pm574/temp/{}".format(fname)
    try: os.remove(path)
    except OSError: pass

if __name__ == "__main__":
    # ------ TRAINING ------ #
    train1()
