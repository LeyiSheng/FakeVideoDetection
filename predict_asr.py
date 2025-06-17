import os
import hydra
import logging
import random

import torch
import torchaudio

from tqdm import tqdm

import numpy as np

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
        fn, length, label, audio_label, video_label = line.split(',')
        if os.path.exists(fn):
            fns.append((fn.strip(), length.strip(), label.strip(), audio_label.strip(), video_label.strip()))
    return fns

def filelist_dfdc(listcsv):
    fns = []
    lines = []
    with open(listcsv) as fp:
        lines = fp.readlines()
    for line in lines:
        fn, length, label, audio_label, video_label = line.split(',')

        fn = fn.replace('/video/', '/audio/')
        fn = fn.replace('.mp4', '.wav')

        if os.path.exists(fn):
            fns.append((fn.strip(), length.strip(), label.strip(), audio_label.strip(), video_label.strip()))
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
        noise_type="pink",
        snr_target=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20, 999999]
        self.noise_type = noise_type
        
        self.noise, sample_rate = torchaudio.load(noise_filename)
        assert sample_rate == 16000
            
    def generate_pink_noise(self, length):
        # pink noise generator, based on numpy
        uneven = length % 2
        X = np.random.randn(length // 2 + 1 + uneven) + 1j * np.random.randn(length // 2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 1.)  # avoid division by zero
        y = (np.fft.irfft(X / S)).real
        if uneven:
            y = y[:-1]
        return torch.from_numpy(y).float().unsqueeze(0) 

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        speech = speech.t()
        if self.noise_type == "pink":
            pink_noise = self.generate_pink_noise(speech.shape[1])
            self.noise = pink_noise
        elif self.noise_type == "white":
            self.noise = torch.randn(speech.size())
            
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
    
    audio_pipeline = torch.nn.Sequential(
        AddNoise(snr_target=cfg.decode.snr_target)
        if cfg.decode.snr_target is not None
        else FunctionalModule(lambda x: x),
        FunctionalModule(
            lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
        ),
    )   
    
    text_transform = TextTransform()
    token_list = text_transform.token_list
    
    model = E2E(len(token_list), cfg.model.audio_backbone).to(device)
    ckpt = torch.load(cfg.audio_ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt)
    model.eval()
    beam_search = get_beam_search_decoder(model, token_list)
    
    fns = filelist(cfg.file_path)
    
    logging.info('loading fns...')
    
    for i, (fn, length, label, al, vl) in enumerate(tqdm(fns)):
        fn = fn.replace('/video/', '/audio/').replace('.mp4', '.wav')
        audio, _ = torchaudio.load(fn, normalize=True)
        audio = audio.transpose(1, 0)
        audio = audio_pipeline(audio)
        
        with torch.no_grad():
            enc_feat, _ = model.encoder(audio.unsqueeze(0).to(device), None)
            enc_feat = enc_feat.squeeze(0)
            nbest_hyps = beam_search(enc_feat)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            predicted = text_transform.post_process(predicted_token_id).replace("<eos>", "")
                
            os.makedirs(os.path.dirname(cfg.asr_infer_path), exist_ok=True)
            with open(cfg.asr_infer_path, 'a') as fp:
                fp.write(f'{fn},{length},{label},{al},{vl},{predicted}\n')
            
    


if __name__ == "__main__":
    main()
