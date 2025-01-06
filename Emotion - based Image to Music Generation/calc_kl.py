#Here I am importing the libraries
import os
import torch
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import argparse

class MetricOutput:
    def __init__(self, mean):
        self.mean = mean

#This is the class for KL divergence calculation
class KLDivergence:
    def __init__(self, use_pca=False, use_activation=False, verbose=False, audio_load_worker=8):
        self.__get_model(use_pca=use_pca, use_activation=use_activation)
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker

    def __get_model(self, use_pca=False, use_activation=False):
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")
        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(
                *list(self.model.embeddings.children())[:-1]
            )
        self.model.eval()

    def get_embeddings_from_audio(self, audio_list, sr=32000):
        embd_lst = []
        try:
            for audio in tqdm(audio_list, disable=(not self.verbose)):
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                embd = self.model.forward(audio, sr)
                if self.model.device == torch.device("cuda"):
                    embd = embd.cpu()
                embd = embd.detach().numpy()
                embd_lst.append(embd)
        except Exception as e:
            print(f"[KL Divergence] get_embeddings_from_audio threw an exception: {str(e)}")
        return np.concatenate(embd_lst, axis=0) if embd_lst else []

    def calculate_distribution(self, embd_lst, bins=100):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        hist, bin_edges = np.histogram(embd_lst, bins=bins, density=True)
        return hist, bin_edges

    def score_from_audio(self, real_audio_list, generated_audio_list):
        try:
            embds_real = self.get_embeddings_from_audio(real_audio_list)
            embds_generated = self.get_embeddings_from_audio(generated_audio_list)

            if len(embds_real) == 0:
                print("[KL Divergence] real audio set is empty, exiting...")
                return MetricOutput(mean=-1)

            if len(embds_generated) == 0:
                print("[KL Divergence] generated audio set is empty, exiting...")
                return MetricOutput(mean=-1)

            hist_real, _ = self.calculate_distribution(embds_real)
            hist_generated, _ = self.calculate_distribution(embds_generated)

            # KL divergence using the integrated approach
            kl_sigmoid, kl_softmax = self.calculate_kl(hist_real, hist_generated)

            return MetricOutput(mean={
                "kullback_leibler_divergence_sigmoid": kl_sigmoid,
                "kullback_leibler_divergence_softmax": kl_softmax
            })

        except Exception as e:
            print(f"[KL Divergence] exception thrown, {str(e)}")
            return MetricOutput(mean=-1)

    def calculate_kl(self, hist_real, hist_generated, eps=1e-6):
        # KL divergence with softmax
        kl_softmax = F.kl_div(
            (torch.tensor(hist_generated).softmax(dim=0) + eps).log(),
            torch.tensor(hist_real).softmax(dim=0),
            reduction="sum",
        )

        # KL divergence with sigmoid
        kl_sigmoid = F.kl_div(
            (torch.tensor(hist_generated).sigmoid() + eps).log(),
            torch.tensor(hist_real).sigmoid(),
            reduction="sum",
        )

        return float(kl_sigmoid), float(kl_softmax)

def find_matching_real_audio(generated_audio_dir, real_audio_dir):
    real_audio_files = []
    generated_audio_files = []

    for gen_file in os.listdir(generated_audio_dir):
        if gen_file.endswith('.wav'):
            gen_file_path = os.path.join(generated_audio_dir, gen_file)
            folder_name = gen_file.split('_')[0]
            # Search for a matching real audio file
            real_file_path = os.path.join(real_audio_dir, folder_name, gen_file)
            if os.path.exists(real_file_path):
                real_audio_files.append(real_file_path)
                generated_audio_files.append(gen_file_path)
            else:
                print(f"Real audio file not found for {gen_file}")

    return real_audio_files, generated_audio_files

def main(real_audio_dir, generated_audio_dir):
    real_audio_files, generated_audio_files = find_matching_real_audio(generated_audio_dir, real_audio_dir)
    
    if not real_audio_files or not generated_audio_files:
        print("No matching audio files found.")
        return

    kl_divergence = KLDivergence(verbose=True)
    result = kl_divergence.score_from_audio(real_audio_files, generated_audio_files)
    print(f"KL Divergence Scores: {result.mean}")


parser = argparse.ArgumentParser()
parser.add_argument('--real_audio_dir', type=str, required=True)
parser.add_argument('--gen_audio_dir', type=str, required=True)
args = parser.parse_args()

main(args.real_audio_dir, args.gen_audio_dir)
