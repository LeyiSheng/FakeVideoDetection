import os
import hydra
import logging
import random
import json

import torch
import torchaudio
import torchvision

from tqdm import tqdm

from datamodule.transforms import TextTransform
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer

from pytorch_lightning import seed_everything

NOISE_FILENAME = './datamodule/babble_noise.wav'


def filelist(listcsv):
    fns = []
    with open(listcsv) as fp:
        lines = fp.readlines()
    for line in lines:
        fn, length, label, audio_label, video_label, target = line.split(',')
        
        fn = fn.replace('/audio/', '/video/')
        fn = fn.replace('.wav', '.mp4')
        
        if os.path.exists(fn):
            fns.append((fn.strip(), length.strip(), label.strip(), audio_label.strip(), video_label.strip(), target.strip()))
    return fns

def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40):
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )

def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())

class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)
    
class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_filename=NOISE_FILENAME,
        snr_target=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20, 999999]
        self.noise, sample_rate = torchaudio.load(noise_filename)
        assert sample_rate == 16000

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        speech = speech.t()
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx : start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.t()




@hydra.main(version_base="1.3", config_path="configs", config_name="df")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()
    device = cfg.device
    
    if cfg.decode.snr_target == 0:
        video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.CenterCrop(88),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize(0.421, 0.165),
        )
    else:
        video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.CenterCrop(88),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.GaussianBlur(5, cfg.decode.snr_target),
            torchvision.transforms.Normalize(0.421, 0.165),
        )
    
    text_transform = TextTransform()
    token_list = text_transform.token_list
    
    model = E2E(len(token_list), cfg.model.visual_backbone).to(device)
    ckpt = torch.load(cfg.video_ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt)
    model.eval()
    beam_search = get_beam_search_decoder(model, token_list)
    
    fns = filelist(cfg.file_path)
    
    logging.info('loading fns...')
    
    total_length = 0
    total_edit_distance = 0
    infos = {}
    
    for i, (fn, length, label, audio_label, video_label, target) in enumerate(tqdm(fns)):
        video = torchvision.io.read_video(fn, pts_unit="sec", output_format="THWC")[0]
        video = video.permute((0, 3, 1, 2)).to(device)
        video = video_pipeline(video)
        
        with torch.no_grad():
            enc_feat, _ = model.encoder(video.unsqueeze(0), None)
            enc_feat = enc_feat.squeeze(0)
            nbest_hyps = beam_search(enc_feat)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            predicted = text_transform.post_process(predicted_token_id).replace("<eos>", "")
                
            err = compute_word_level_distance(target, predicted)
            length = len(target.split())
            if length == 0:
                wer = 0
            else:
                wer = err / length
            
            total_edit_distance += err
            total_length += length
            
            info = {
                'fn': fn,
                'label': label,
                'audio label': audio_label,
                'video label': video_label,
                'predicted': predicted,
                'target': target,
                'wer': wer,
            }
                
            json.dump(info, open(cfg.infer_path, 'a'), indent=4, ensure_ascii=False)
            
            infos[fn] = info
            
        del video, enc_feat, predicted_token_id
        torch.cuda.empty_cache()
    
    print(f'total wer: {total_edit_distance / total_length}')
    json.dump(infos, open(cfg.infer_path, 'w'), indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    main()
