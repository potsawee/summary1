import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import random
from datetime import datetime
from collections import OrderedDict

from data_meeting import TopicSegment, Utterance, bert_tokenizer, DA_MAPPING
from data import cnndm
from data.cnndm import ProcessedDocument, ProcessedSummary
from models.hierarchical_rnn_v5 import EncoderDecoder, DALabeller, EXTLabeller
from models.neural import LabelSmoothingLoss

def train_v5():
    print("Start training hierarchical RNN model")
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['use_gpu']        = True
    args['num_utterances'] = 1500  # max no. utterance in a meeting
    args['num_words']      = 64    # max no. words in an utterance
    args['summary_length'] = 300   # max no. words in a summary
    args['summary_type']   = 'short'   # long or short summary
    args['vocab_size']     = 30522 # BERT tokenizer
    args['embedding_dim']   = 256   # word embeeding dimension
    args['rnn_hidden_size'] = 512 # RNN hidden size

    args['dropout']        = 0.1
    args['num_layers_enc'] = 2    # in total it's num_layers_enc*2 (word/utt)
    args['num_layers_dec'] = 1

    args['batch_size']      = 1
    args['update_nbatches'] = 2
    args['num_epochs']      = 20
    args['random_seed']     = 597
    args['best_val_loss']     = 1e+10
    args['val_batch_size']    = 1 # 1 for now --- evaluate ROUGE
    args['val_stop_training'] = 5

    args['lr']         = 1.0
    args['adjust_lr']  = True     # if True overwrite the learning rate above
    args['initial_lr'] = 0.01       # lr = lr_0*step^(-decay_rate)
    args['decay_rate'] = 0.5
    args['label_smoothing'] = 0.1

    args['a_ll']  = 0.1
    args['a_da']  = 0.0
    args['a_ext'] = 1.0

    args['memory_utt'] = False

    args['model_save_dir'] = "/home/alta/summary/pm574/summariser1/lib/trained_models2/"
    args['load_model'] = "/home/alta/summary/pm574/summariser1/lib/trained_models2/model-HGRUV5_CNNDM_FEB26A-ep12-bn0" # add .pt later
    # args['load_model'] = "/home/alta/summary/pm574/summariser1/lib/trained_models2/model-HGRUV5_FEB28A-ep6"
    # args['load_model'] = "/home/alta/summary/pm574/summariser1/lib/trained_models2/model-HGRUV5MEM_APR8A-ep1"
    # args['load_model'] = None
    args['model_name'] = 'HGRUV5_APR17ML44'
    # ---------------------------------------------------------------------------------- #
    print_config(args)

    if args['use_gpu']:
        if 'X_SGE_CUDA_DEVICE' in os.environ: # to run on CUED stack machine
            print('running on the stack... 1 GPU')
            cuda_device = os.environ['X_SGE_CUDA_DEVICE']
            print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
        else:
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
    # make the training data 100
    random.shuffle(valid_data)
    train_data.extend(valid_data[:6])
    valid_data = valid_data[6:]

    model = EncoderDecoder(args, device=device)
    print(model)
    NUM_DA_TYPES = len(DA_MAPPING)
    da_labeller = DALabeller(args['rnn_hidden_size'], NUM_DA_TYPES, device)
    print(da_labeller)
    ext_labeller = EXTLabeller(args['rnn_hidden_size'], device)
    print(ext_labeller)

    # Load model if specified (path to pytorch .pt)
    if args['load_model'] != None:
        model_path = args['load_model'] + '.pt'
        try:
            model.load_state_dict(torch.load(model_path))
        except RuntimeError: # need to remove module
            # Main model
            model_state_dict = torch.load(model_path)
            new_model_state_dict = OrderedDict()
            for key in model_state_dict.keys():
                new_model_state_dict[key.replace("module.","")] = model_state_dict[key]

            if args['memory_utt']:
                model.load_state_dict(new_model_state_dict, strict=False)
            else:
                model.load_state_dict(new_model_state_dict)

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

    da_criterion = nn.NLLLoss(reduction='none')
    ext_criterion = nn.BCELoss(reduction='none')

    optimizer = optim.Adam(model.parameters(),lr=args['lr'],betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer.zero_grad()

    # DA labeller optimiser
    da_optimizer = optim.Adam(da_labeller.parameters(),lr=args['lr'],betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    da_optimizer.zero_grad()

    # extractive labeller optimiser
    ext_optimizer = optim.Adam(ext_labeller.parameters(),lr=args['lr'],betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    ext_optimizer.zero_grad()

    # validation losses
    best_val_loss = args['best_val_loss']
    best_epoch    = 0
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

            input, u_len, w_len, target, tgt_len, _, dialogue_acts, extractive_label = get_a_batch(
                    train_data, idx, BATCH_SIZE,
                    args['num_utterances'], args['num_words'],
                    args['summary_length'], args['summary_type'], device)

            # decoder target
            decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device, mask_offset=True)
            decoder_target = decoder_target.view(-1)
            decoder_mask = decoder_mask.view(-1)

            decoder_output, u_output, attn_scores, cov_scores, u_attn_scores = model(input, u_len, w_len, target)

            loss = criterion(decoder_output.view(-1, args['vocab_size']), decoder_target)
            loss = (loss * decoder_mask).sum() / decoder_mask.sum()

            # multitask(2): dialogue act prediction
            da_output = da_labeller(u_output)
            loss_utt_mask = length2mask(u_len, BATCH_SIZE, args['num_utterances'], device)
            loss_da = da_criterion(da_output.view(-1, NUM_DA_TYPES), dialogue_acts.view(-1)).view(BATCH_SIZE, -1)
            loss_da = (loss_da * loss_utt_mask).sum() / loss_utt_mask.sum()

            # multitask(3): extractive label prediction
            ext_output = ext_labeller(u_output).squeeze(-1)
            loss_ext = ext_criterion(ext_output, extractive_label)
            loss_ext = (loss_ext * loss_utt_mask).sum() / loss_utt_mask.sum()

            total_loss = args['a_ll']*loss + args['a_da']*loss_da + args['a_ext']*loss_ext

            total_loss.backward()

            idx += BATCH_SIZE

            if bn % args['update_nbatches'] == 0:
                # gradient_clipping
                max_norm = 0.5
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                nn.utils.clip_grad_norm_(da_labeller.parameters(), max_norm)
                nn.utils.clip_grad_norm_(ext_labeller.parameters(), max_norm)
                # update the gradients
                if args['adjust_lr']:
                    adjust_lr(optimizer, args['initial_lr'], args['decay_rate'], training_step)
                    adjust_lr(da_optimizer, args['initial_lr'], args['decay_rate'], training_step)
                    adjust_lr(ext_optimizer, args['initial_lr'], args['decay_rate'], training_step)
                optimizer.step()
                optimizer.zero_grad()
                da_optimizer.step()
                da_optimizer.zero_grad()
                ext_optimizer.step()
                ext_optimizer.zero_grad()
                training_step += args['batch_size']*args['update_nbatches']

            if bn % 1 == 0:
                print("[{}] batch {}/{}: loss = {:.5f} | loss_da = {:.5f} | loss_ext = {:.5f}".
                    format(str(datetime.now()), bn, num_batches, loss, loss_da, loss_ext))
                sys.stdout.flush()

            if bn == 0: # e.g. eval every epoch
                # ---------------- Evaluate the model on validation data ---------------- #
                print("Evaluating the model at epoch {} step {}".format(epoch, bn))
                print("learning_rate = {}".format(optimizer.param_groups[0]['lr']))
                # switch to evaluation mode
                model.eval()
                da_labeller.eval()
                ext_labeller.eval()
                with torch.no_grad():
                    avg_val_loss = evaluate(model, da_labeller, ext_labeller, NUM_DA_TYPES,
                                            valid_data, VAL_BATCH_SIZE, args, device,
                                            args['a_ll'], args['a_da'], args['a_ext'])
                print("avg_val_loss_per_token = {}".format(avg_val_loss))
                # switch to training mode
                model.train()
                da_labeller.train()
                ext_labeller.train()
                # ------------------- Save the model OR Stop training ------------------- #
                state = {
                    'epoch': epoch, 'bn': bn,
                    'training_step': training_step,
                    'model': model.state_dict(),
                    'da_labeller': da_labeller.state_dict(),
                    'ext_labeller': ext_labeller.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }
                if avg_val_loss < best_val_loss:
                    stop_counter = 0
                    best_val_loss = avg_val_loss
                    best_epoch = epoch

                    savepath = args['model_save_dir']+"model-{}-ep{}.pt".format(args['model_name'], 999) # 999 = best
                    torch.save(state, savepath)
                    print("Model improved & saved at {}".format(savepath))
                else:
                    print("Model not improved #{}".format(stop_counter))
                    savepath = args['model_save_dir']+"model-{}-ep{}.pt".format(args['model_name'], 000) # 000 = current
                    torch.save(state, savepath)
                    print("Model NOT improved & saved at {}".format(savepath))
                    if stop_counter < VAL_STOP_TRAINING:
                        print("Just continue training ---- no loading old weights")
                        stop_counter += 1
                    else:
                        print("Model has not improved for {} times! Stop training.".format(VAL_STOP_TRAINING))
                        return

    print("End of training hierarchical RNN model")

def length2mask(length, batch_size, max_len, device):
    mask = torch.zeros((batch_size, max_len), dtype=torch.float)
    for bn in range(batch_size):
        l = length[bn].item()
        mask[bn,:l].fill_(1.0)
    mask = mask.to(device)
    return mask

def evaluate(model, da_labeller, ext_labeller, num_da_types,
            eval_data, eval_batch_size, args, device, a_ll, a_da, a_ext):
    # num_eval_epochs = int(eval_data['num_data']/eval_batch_size) + 1
    num_eval_epochs = int(len(eval_data)/eval_batch_size)

    print("num_eval_epochs = {}".format(num_eval_epochs))
    eval_idx = 0
    eval_total_loss = 0.0
    eval_total_tokens = 0

    eval_loss_da  = 0.0
    eval_loss_ext = 0.0

    criterion = nn.NLLLoss(reduction='none')
    da_criterion = nn.NLLLoss(reduction='none')
    ext_criterion = nn.BCELoss(reduction='none')

    for bn in range(num_eval_epochs):

        input, u_len, w_len, target, tgt_len, _, dialogue_acts, extractive_label = get_a_batch(
                eval_data, eval_idx, eval_batch_size,
                args['num_utterances'], args['num_words'],
                args['summary_length'], args['summary_type'], device)

        # decoder target
        decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device)
        decoder_target = decoder_target.view(-1)
        decoder_mask = decoder_mask.view(-1)

        decoder_output, u_output, attn_scores, cov_scores, u_attn_scores = model(input, u_len, w_len, target)

        loss = criterion(decoder_output.view(-1, args['vocab_size']), decoder_target)
        eval_total_loss += (loss * decoder_mask).sum().item()
        eval_total_tokens += decoder_mask.sum().item()

        # multitask(2): dialogue act prediction
        da_output = da_labeller(u_output)
        loss_utt_mask = length2mask(u_len, eval_batch_size, args['num_utterances'], device)
        loss_da = da_criterion(da_output.view(-1, num_da_types), dialogue_acts.view(-1)).view(eval_batch_size, -1)
        loss_da = (loss_da * loss_utt_mask).sum() / loss_utt_mask.sum()

        # multitask(3): extractive label prediction
        ext_output = ext_labeller(u_output).squeeze(-1)
        loss_ext = ext_criterion(ext_output, extractive_label)
        loss_ext = (loss_ext * loss_utt_mask).sum() / loss_utt_mask.sum()

        eval_loss_da  += loss_da.cpu().item()
        eval_loss_ext += loss_ext.cpu().item()

        eval_idx += eval_batch_size

        print("#", end="")
        sys.stdout.flush()
    print()

    avg_eval_loss = eval_total_loss / eval_total_tokens
    eval_loss_da  /= num_eval_epochs
    eval_loss_ext /= num_eval_epochs

    avg_all_loss = a_ll*avg_eval_loss + a_da*eval_loss_da + a_ext*eval_loss_ext

    return avg_all_loss

def adjust_lr(optimizer, lr0, decay_rate, step):
    """to adjust the learning rate for both encoder & decoder --- DECAY"""
    step = step + 1 # plus 1 to avoid ZeroDivisionError

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

    # topic boundaries
    topic_boundary_label = torch.zeros((batch_size, num_utterances), dtype=torch.float)

    # dialogue act
    dialogue_acts = torch.zeros((batch_size, num_utterances), dtype=torch.long)

    # extractive label
    extractive_label = torch.zeros((batch_size, num_utterances), dtype=torch.float)

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
                dialogue_acts[bn,utt_id] = DA_MAPPING[utterance.dialogueact]
                extractive_label[bn,utt_id] = utterance.extsum_label
                utt_id += 1
                if utt_id == num_utterances: break

            topic_boundary_label[bn, utt_id-1] = 1

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
    topic_boundary_label = topic_boundary_label.to(device)
    dialogue_acts = dialogue_acts.to(device)
    extractive_label = extractive_label.to(device)

    # covert numpy to torch tensor (for multiple GPUs purpose)
    utt_lengths = torch.from_numpy(utt_lengths)
    word_lengths = torch.from_numpy(word_lengths)
    summary_lengths = torch.from_numpy(summary_lengths)

    return input, utt_lengths, word_lengths, summary, summary_lengths, topic_boundary_label, dialogue_acts, extractive_label

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

if __name__ == "__main__":
    # ------ TRAINING ------ #
    train_v5()
