import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import random
from datetime import datetime

from train1 import load_ami_data, get_a_batch, shift_decoder_target, print_config
from data_meeting import TopicSegment, Utterance, bert_tokenizer
from models.hierarchical_rnn_v2 import EncoderDecoder
from models.neural import LabelSmoothingLoss

class AdaptiveBias(nn.Module):
    def __init__(self, hidden_size, init_bias, device):
        super(AdaptiveBias, self).__init__()
        self.device = device
        # self.weight = nn.Parameter(torch.ones(1, hidden_size)*init_bias, requires_grad=True)
        self.linear1 = nn.Linear(hidden_size, 64, bias=True)
        self.linear2 = nn.Linear(64, hidden_size, bias=True)

        self.to(device)

    def param_init(self):
        for name, p in self.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                if 'bias' in name: nn.init.zeros_(p)

    def forward(self, cov_y):
        # cov_y => batch_size * vocab_size
        # return self.weight * cov_y
        return self.linear2(nn.functional.relu(self.linear1(cov_y)))


def train_adaptive_bias():
    print("Start training adaptive bias")
    # ---------------------------------------------------------------------------------- #
    args = {}
    args['load_model'] = "/home/alta/summary/pm574/summariser1/lib/trained_models/model-HGRUV2_CNNDM_AMI_JAN24A-ep17.pt"
    args['num_utterances']  = 2000  # max no. utterance in a meeting
    args['num_words']       = 64    # max no. words in an utterance
    args['summary_length']  = 800   # max no. words in a summary
    args['summary_type']    = 'long'   # long or short summary
    args['vocab_size']      = 30522 # BERT tokenizer
    args['dropout']         = 0.0
    args['embedding_dim']   = 256   # word embeeding dimension
    args['rnn_hidden_size'] = 512 # RNN hidden size
    args['num_layers_enc']  = 1    # in total it's num_layers_enc*3 (word/utt/seg)
    args['num_layers_dec']  = 1

    args['init_bias']       = 20
    args['random_seed']     = 28
    args['lr']              = 1.0

    args['abias_save_dir'] = "/home/alta/summary/pm574/summariser1/lib/adaptive_bias_weights/"
    args['abias_name']     = "ABIAS_FEB21A2"
    # ---------------------------------------------------------------------------------- #
    print_config(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = '1' # choose the device (GPU) here
    device = 'cuda'

    train_data = load_ami_data('train')
    valid_data = load_ami_data('valid')

    random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])

    adaptivebias = AdaptiveBias(args['vocab_size'], args['init_bias'], device)
    print(adaptivebias)

    model = EncoderDecoder(args, device)
    model_path = args['load_model']
    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError: # need to remove module
        # Main model
        model_state_dict = torch.load(model_path)
        new_model_state_dict = OrderedDict()
        for key in model_state_dict.keys():
            new_model_state_dict[key.replace("module.","")] = model_state_dict[key]
        model.load_state_dict(new_model_state_dict)
    model.eval()

    for p in model.parameters(): p.requires_grad = False
    
    optimizer = optim.Adam(adaptivebias.parameters(), lr=args['lr'] , betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    optimizer.zero_grad()

    criterion = LabelSmoothingLoss(num_classes=args['vocab_size'], smoothing=0.1, reduction='none')

    batch_size = 1
    num_epochs = 5
    time_step  = args['summary_length']
    vocab_size = args['vocab_size']
    start_token_id = 101 # [CLS]
    stop_token_id  = 103 # [MASK]

    for epoch in range(num_epochs):
        print("======================= Training epoch {} =======================".format(epoch))
        num_train_data = len(train_data)
        num_batches = int(num_train_data/batch_size)
        print("num_batches = {}".format(num_batches))
        random.shuffle(train_data)

        idx = 0

        for bn in range(num_batches):
            input, u_len, w_len, target, tgt_len, _, _, _ = get_a_batch(
                    train_data, idx, batch_size,
                    args['num_utterances'], args['num_words'],
                    args['summary_length'], args['summary_type'], device)

            # decoder target
            decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device, mask_offset=False)
            decoder_target = decoder_target.view(-1)
            decoder_mask = decoder_mask.view(-1)

            enc_output_dict = model.encoder(input, u_len, w_len)

            y_out  = torch.ones((batch_size, time_step, vocab_size), dtype=torch.float).to(device)
            y_pred = torch.zeros((batch_size, time_step), dtype=torch.long).to(device)
            y_pred.fill_(stop_token_id)

            y_init = torch.zeros((batch_size, 1), dtype=torch.long).to(device)
            y_init.fill_(start_token_id)

            ht = model.decoder.init_h0(batch_size)

            for t in range(time_step-1):
                # we have already obtained prediction up to time step 't' and want to predict 't+1'
                if t == 0:
                    decoder_output, ht = model.decoder.forward_step(y_init, ht, enc_output_dict, logsoftmax=False)
                    output = decoder_output
                else:
                    decoder_output, ht = model.decoder.forward_step(y_pred[:, t-1].unsqueeze(-1), ht, enc_output_dict, logsoftmax=False)
                    # sum y_out from 0 upto t-1
                    cov_y =  y_out[:, :t, :].sum(dim=1)
                    # normalise cov_y
                    cov_y = cov_y / cov_y.sum(dim=-1).unsqueeze(-1)
                    bias = adaptivebias(cov_y)
                    # maybe think about in what domain we should add this bias?? LogSoftmax??
                    output = decoder_output - bias
                y_out[:, t, :] = nn.functional.softmax(output, dim=-1)
                y_pred[:, t]   = output.argmax(dim=-1)

                # if t % 100 == 0: print("t = {}".format(t))
                #### ONLY WORKS WITH batch_size = 1
                if y_pred[0, t] == stop_token_id: break

            log_y_out = torch.log(y_out)
            loss = criterion(log_y_out.view(-1, args['vocab_size']), decoder_target)
            loss = (loss * decoder_mask).sum() / decoder_mask.sum()

            loss.backward()

            print("[{}] batch {}/{}: loss = {:5f}".format(str(datetime.now()), bn, num_batches, loss))
            sys.stdout.flush()

            idx += batch_size

            optimizer.step()
            optimizer.zero_grad()

            if bn % 20 == 0:
                print("======================== GENERATED SUMMARY ========================")
                print(bert_tokenizer.decode(y_pred[0].cpu().numpy()[:tgt_len[0]]))
                print("======================== REFERENCE SUMMARY ========================")
                print(bert_tokenizer.decode(decoder_target.view(batch_size,args['summary_length'])[0,:tgt_len[0]].cpu().numpy()))

        print("END OF EPOCH {}".format(epoch))

        # Evaluation
        with torch.no_grad():
            avg_val_loss = eval_model_with_bias(model, adaptivebias, valid_data, args, device)
        print("avg_val_loss_per_token = {}".format(avg_val_loss))

        # Save the model
        savepath = args['abias_save_dir']+"abias-{}-ep{}.pt".format(args['abias_name'],epoch+1)
        torch.save(adaptivebias.state_dict(), savepath)

def eval_model_with_bias(model, adaptivebias, eval_data, args, device):
    num_eval_epochs = len(eval_data)
    from rouge import Rouge
    rouge = Rouge()
    bert_decoded_outputs = []
    bert_decoded_targets = []

    time_step  = args['summary_length']
    vocab_size = args['vocab_size']
    start_token_id = 101 # [CLS]
    stop_token_id  = 103 # [MASK]

    eval_idx = 0

    for bn in range(num_eval_epochs):

        input, u_len, w_len, target, tgt_len, _, _, _ = get_a_batch(
                eval_data, eval_idx, 1,
                args['num_utterances'], args['num_words'],
                args['summary_length'], args['summary_type'], device)

        # decoder target
        decoder_target, decoder_mask = shift_decoder_target(target, tgt_len, device)
        decoder_target = decoder_target.view(-1)
        decoder_mask = decoder_mask.view(-1)

        enc_output_dict = model.encoder(input, u_len, w_len)

        y_out  = torch.ones((1, time_step, vocab_size), dtype=torch.float).to(device)
        y_pred = torch.zeros((1, time_step), dtype=torch.long).to(device)
        y_pred.fill_(stop_token_id)

        y_init = torch.zeros((1, 1), dtype=torch.long).to(device)
        y_init.fill_(start_token_id)

        ht = model.decoder.init_h0(1)

        for t in range(time_step-1):
            # we have already obtained prediction up to time step 't' and want to predict 't+1'
            if t == 0:
                decoder_output, ht = model.decoder.forward_step(y_init, ht, enc_output_dict, logsoftmax=False)
                output = decoder_output
            else:
                decoder_output, ht = model.decoder.forward_step(y_pred[:, t-1].unsqueeze(-1), ht, enc_output_dict, logsoftmax=False)
                # sum y_out from 0 upto t-1
                cov_y =  y_out[:, :t, :].sum(dim=1)
                # normalise cov_y
                cov_y = cov_y / cov_y.sum(dim=-1).unsqueeze(-1)
                bias = adaptivebias(cov_y)
                # maybe think about in what domain we should add this bias?? LogSoftmax??
                output = decoder_output - bias
            y_out[:, t, :] = nn.functional.softmax(output, dim=-1)
            y_pred[:, t]   = output.argmax(dim=-1)

            if y_pred[0, t] == stop_token_id: break


        bert_decoded_output = bert_tokenizer.decode(y_pred[0].cpu().numpy())
        stop_idx = bert_decoded_output.find('[MASK]')
        bert_decoded_output = bert_decoded_output[:stop_idx]
        bert_decoded_outputs.append(bert_decoded_output)

        bert_decoded_target = bert_tokenizer.decode(decoder_target.cpu().numpy())
        stop_idx2 = bert_decoded_target.find('[MASK]')
        bert_decoded_target = bert_decoded_target[:stop_idx2]
        bert_decoded_targets.append(bert_decoded_target)

        eval_idx += 1

        print("#", end="")
        sys.stdout.flush()

    print()

    try:
        scores = rouge.get_scores(bert_decoded_outputs, bert_decoded_targets, avg=True)
        return (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f'])*(-100)/3
    except ValueError:
        return 0


train_adaptive_bias()
