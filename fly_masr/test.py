import argparse
import pickle
import torch
from tqdm import tqdm
from wav_time_count import get_duration_wav
from config.conf import pickle_file, device, input_dim, LFR_m, LFR_n, sos_id, eos_id
from data.data_process import build_LFR_features
from features.feature_generate import extract_feature
from utils.util import cer_function

parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Decoding.")
# decode
parser.add_argument('--beam_size', default=5, type=int,
                    help='Beam size')
parser.add_argument('--nbest', default=1, type=int,
                    help='Nbest size')
parser.add_argument('--decode_max_len', default=100, type=int,
                    help='Max output length. If ==0 (default), it uses a '
                         'end-detect function to automatically find maximum '
                         'hypothesis lengths')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    char_list = data['IVOCAB']
    samples = data['test']
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model = checkpoint['model'].to(device)
    model.eval()
    num_samples = len(samples)
    total_cer = 0
    RTF = 0

    for i in tqdm(range(num_samples)):
        sample = samples[i]
        wave = sample['wave']
        trn = sample['trn']

        atime = get_duration_wav(wave)

        feature = extract_feature(input_file=wave, feature='fbank', dim=input_dim, cmvn=True)
        feature = build_LFR_features(feature, m=LFR_m, n=LFR_n)
        # feature = np.expand_dims(feature, axis=0)
        input = torch.from_numpy(feature).to(device)
        input_length = [input.shape[0]]
        input_length = torch.LongTensor(input_length).to(device)


        with torch.no_grad():
            nbest_hyps , run_time = model.recognize(input, input_length.to('cpu'), char_list, args)


        hyp_list = []
        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [char_list[idx] for idx in out if idx not in (sos_id, eos_id)]
            out = ''.join(out)
            hyp_list.append(out)

        print(hyp_list)

        gt = [char_list[idx] for idx in trn if idx not in (sos_id, eos_id)]
        gt = ''.join(gt)
        gt_list = [gt]

        print(gt_list)

        cer = cer_function(gt_list, hyp_list)
        total_cer += cer
        aRTF = run_time / atime
        RTF += aRTF


    avg_RTF = RTF / num_samples
    avg_cer = total_cer / num_samples

    print('Average CER: ' + str(avg_cer))
    print(f'Average RTF:{avg_RTF}')
