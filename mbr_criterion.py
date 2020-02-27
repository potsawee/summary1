import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from data_meeting import bert_tokenizer

from rouge import Rouge
rouge = Rouge()

START_TOKEN_ID = 101 # [CLS]
SEP_TOKEN_ID   = 102 # [SEP]
STOP_TOKEN_ID  = 103 # [MASK]
VOCAB_SIZE     = 30522

def grad_sampling(model, input, u_len, w_len, target,
                num_samples, lambda1, device, printout):

    total_r1 = 0
    total_r2 = 0
    total_rl = 0

    batch_size = input.size(0)
    if batch_size != 1: raise ValueError("batch_size error")

    # tgt -> reference (y)
    reference = bert_tokenizer.decode(target[0].cpu().numpy())
    stop_idx = reference.find('[SEP]')
    reference = reference[:stop_idx]
    time_step = len(bert_tokenizer.encode(reference)) + 1 # plus 1 just in case

    if printout:
        print("reference: {}".format(reference))
        print("-----------------------------------------------------------------------------------------")

    grads   = [None for _ in range(num_samples)]
    metrics = [None for _ in range(num_samples)]

    for i in range(num_samples):
        # forward-pass ENCODER --- need to do forward pass again as autograd freed up memory
        enc_output_dict = model.encoder(input, u_len, w_len) # memory
        u_output = enc_output_dict['u_output']

        # forward-pass DECODER
        xt = torch.zeros((batch_size, 1), dtype=torch.int64).to(device)
        xt.fill_(START_TOKEN_ID) # 101

        # initial hidden state
        ht = torch.zeros((model.decoder.num_layers, batch_size, model.decoder.dec_hidden_size),
                                    dtype=torch.float).to(device)
        for bn, l in enumerate(u_len): ht[:,bn,:] = u_output[bn,l-1,:].unsqueeze(0)

        log_prob_seq = 0
        generated_tokens = []
        for t in range(time_step-1):
            decoder_output, ht, _ = model.decoder.forward_step(xt, ht, enc_output_dict, logsoftmax=False)
            output_prob = F.softmax(decoder_output, dim=-1)

            m = Categorical(output_prob)
            sample = m.sample()
            log_prob_t = m.log_prob(sample)

            xt = sample.unsqueeze(-1)
            log_prob_seq += log_prob_t
            token = sample.item()
            generated_tokens.append(token)

            if token == SEP_TOKEN_ID: break

        # generated_tokens -> hypothesis (y_hat)
        hypothesis = bert_tokenizer.decode(generated_tokens)
        stop_idx = hypothesis.find('[SEP]')
        if stop_idx != -1: hypothesis = hypothesis[:stop_idx]

        # Compute D(y, y_hat)
        try:
            scores = rouge.get_scores(hypothesis, reference)
            r1 = scores[0]['rouge-1']['f']
            r2 = scores[0]['rouge-2']['f']
            rl = scores[0]['rouge-l']['f']
        except ValueError:
            r1 = 0
            r2 = 0
            r3 = 0

        metric = -1 * (r1+r2+rl) # since we 'minimise' the criterion

        if printout:
            print("sample{} [{:.2f}]: {}".format(i, -100*metric, hypothesis))
            print("-----------------------------------------------------------------------------------------")


        total_r1 += r1
        total_r2 += r2
        total_rl += rl

        # scale to gradient by metric
        # log_prob_seq *= metric
        # log_prob_seq *= lambda1
        # log_prob_seq.backward()

        # len(grad) = the number of model.parameters() --- checked!
        grad = torch.autograd.grad(log_prob_seq, model.parameters())
        grads[i] = grad
        metrics[i] = metric

    mean_x  = sum(metrics) / len(metrics)
    metrics = [xi - mean_x for xi in metrics]

    # for param in model.parameters(): param.grad /= num_samples

    for i in range(num_samples):
        for n, param in enumerate(model.parameters()):
            if i == 0:
                param.grad  = metrics[i] * grads[i][n]
            else:
                param.grad += metrics[i] * grads[i][n]

    for param in model.parameters(): param.grad *= lambda1 / num_samples

    r1_avg = total_r1 / num_samples
    r2_avg = total_r2 / num_samples
    rl_avg = total_rl / num_samples

    return r1_avg, r2_avg, rl_avg

def beamsearch_approx(model, input, u_len, w_len, target,
                beam_width, device, printout):

    vocab_size = VOCAB_SIZE

    total_r1 = 0
    total_r2 = 0
    total_rl = 0

    batch_size = input.size(0)
    if batch_size != 1: raise ValueError("batch_size error")

    # tgt -> reference (y)
    reference = bert_tokenizer.decode(target[0].cpu().numpy())
    stop_idx = reference.find('[SEP]')
    reference = reference[:stop_idx]
    time_step = len(bert_tokenizer.encode(reference)) + 1 # plus 1 just in case

    if printout:
        print("reference: {}".format(reference))
        print("-----------------------------------------------------------------------------------------")

    # forward-pass ENCODER --- need to do forward pass again as autograd freed up memory
    enc_output_dict = model.encoder(input, u_len, w_len) # memory
    u_output = enc_output_dict['u_output']

    # initial hidden state
    ht = torch.zeros((model.decoder.num_layers, batch_size, model.decoder.dec_hidden_size),
                                dtype=torch.float).to(device)
    for bn, l in enumerate(u_len): ht[:,bn,:] = u_output[bn,l-1,:].unsqueeze(0)
    beam_ht = [None for _ in range(beam_width)]
    for _k in range(beam_width): beam_ht[_k] = ht.clone()

    # beam xt
    beam_xt = [None for _ in range(beam_width)]
    for i in range(beam_width):
        xt = torch.zeros((batch_size, 1), dtype=torch.int64).to(device)
        xt.fill_(START_TOKEN_ID) # 101
        beam_xt[i] = xt

    beam_scores = [0.0 for _ in range(beam_width)]
    beam_generated_tokens = [[] for _ in range(beam_width)]

    for t in range(time_step-1):
        decoder_output_t_array = torch.zeros((batch_size, beam_width*vocab_size))
        temp_ht = [None for _ in range(beam_width)]
        for i in range(beam_width):
            decoder_output, temp_ht[i], _ = model.decoder.forward_step(beam_xt[i], beam_ht[i], enc_output_dict, logsoftmax=True)
            decoder_output_t_array[0, i*vocab_size:(i+1)*vocab_size] = decoder_output
            decoder_output_t_array[0, i*vocab_size:(i+1)*vocab_size] += beam_scores[i]

            if t == 0:
                decoder_output_t_array[0,(i+1)*vocab_size:] = float('-inf')
                break

        topk_scores, topk_ids = torch.topk(decoder_output_t_array, beam_width, dim=-1)

        scores = topk_scores[0]
        indices = topk_ids[0]

        new_beams_scores = [None for _ in range(beam_width)]
        new_beam_generated_tokens = [None for _ in range(beam_width)]

        for i in range(beam_width):
            vocab_idx = indices[i] % vocab_size
            beam_idx  = int(indices[i] / vocab_size)

            new_beams_scores[i] = scores[i]
            new_beam_generated_tokens[i] = list(beam_generated_tokens[beam_idx])
            new_beam_generated_tokens[i].append(vocab_idx.item())

            beam_ht[i] = temp_ht[beam_idx]
            xt = torch.zeros((batch_size, 1), dtype=torch.int64).to(device)
            xt.fill_(vocab_idx) # 101
            beam_xt[i] = xt

        beam_scores = new_beams_scores
        beam_generated_tokens = new_beam_generated_tokens

        if t % 10 == 0: print("#", end="")
        sys.stdout.flush()
    print()
    # print("=========================  t = {} =========================".format(t))
    # for ik in range(beam_width):
        # print("beam{}: [{:.5f}]".format(ik, beam_scores[ik]),bert_tokenizer.
                        # decode(beam_generated_tokens[ik]))

    # Normalise the probablilty
    sum_prob = 0.0
    for i in range(beam_width): sum_prob += torch.exp(beam_scores[i])
    norm_probs = [None for _ in range(beam_width)]
    for i in range(beam_width):
        norm_probs[i] = torch.exp(beam_scores[i]) / sum_prob


    for i in range(beam_width):
        generated_tokens = beam_generated_tokens[i]

        # generated_tokens -> hypothesis (y_hat)
        hypothesis = bert_tokenizer.decode(generated_tokens)
        stop_idx = hypothesis.find('[SEP]')
        if stop_idx != -1: hypothesis = hypothesis[:stop_idx]

        if printout:
            print("beam{}: {}".format(i, hypothesis))
            print("-----------------------------------------------------------------------------------------")

        # Compute D(y, y_hat)
        scores = rouge.get_scores(hypothesis, reference)
        r1 = scores[0]['rouge-1']['f']
        r2 = scores[0]['rouge-2']['f']
        rl = scores[0]['rouge-l']['f']

        metric = -1 * (r1+r2+rl) # since we 'minimise' the criterion

        total_r1 += r1
        total_r2 += r2
        total_rl += rl

        # scale to gradient by metric
        this_loss = norm_probs[i] * metric
        this_loss.backward(retain_graph=True)

    r1_avg = total_r1 / beam_width
    r2_avg = total_r2 / beam_width
    rl_avg = total_rl / beam_width

    return r1_avg, r2_avg, rl_avg
