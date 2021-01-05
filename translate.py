''' Translate input text with trained model. '''

import torch
import argparse
import joblib as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Translator import Translator
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from vocabulary import Vocabulary


def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    data = pickle.load(open(opt.data_pkl, 'rb'))
    src_vocb, trg_vocab = data['vocab']['src'], data['vocab']['trg']
    opt.src_pad_idx = src_vocb.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = trg_vocab.stoi[Constants.PAD_WORD]
    opt.trg_bos_idx = trg_vocab.stoi[Constants.BOS_WORD]
    opt.trg_eos_idx = trg_vocab.stoi[Constants.EOS_WORD]

    test_inputs = torch.tensor(data['test']['src'])
    test_outputs = torch.tensor(data['test']['trg'])
    test_data = TensorDataset(test_inputs, test_outputs)
    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=1)
    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    translator = Translator(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).to(device)

    with open(opt.output, 'w') as f:
        for example in tqdm(test_data_loader, mininterval=2, desc='  - (Test)', leave=False):
            src_seq = example[0]

            pred_seq = translator.translate_sentence(src_seq).to(device)
            pred_line = ' '.join(trg_vocab.itos[idx] for idx in pred_seq)
            pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '').strip()
            pred_line = 'Predicted: ' + pred_line

            trg_seq = example[1].detach().cpu().numpy()
            trg_line = ' '.join(trg_vocab.itos[idx] for idx in trg_seq)
            trg_line = trg_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '').\
                replace(Constants.PAD_WORD, '').strip()
            trg_line = 'Ground truth: ' + trg_line

            line = '\n'.join([pred_line, trg_line])
            f.write(line + '\n\n')

    print('[Info] Finished.')

if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data m30k_deen_shr.pkl -no_cuda
    '''
    main()
