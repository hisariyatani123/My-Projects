#Here I am importing the libraries
import os
import torch
import numpy as np
from scipy import linalg
from tqdm import tqdm
from torch import nn
import argparse

class MetricOutput:
    def __init__(self, mean):
        self.mean = mean

#This class is for FAD calculation
class FrechetAudioDistance:
    def __init__(self, use_pca=False, use_activation=False, verbose=False, audio_load_worker=8):
        self.__get_model(use_pca=use_pca, use_activation=use_activation)
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker
    
    #Here I am using vggish model so I a loading the model here and putting it on eval mode
    def __get_model(self, use_pca=False, use_activation=False):
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")
        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(
                *list(self.model.embeddings.children())[:-1]
            )
        self.model.eval()
    
    #Here I am getting embeeding from audio 
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
            print(f"[Frechet Audio Distance] get_embeddings_from_audio threw an exception: {str(e)}")
        return np.concatenate(embd_lst, axis=0) if embd_lst else []
    
    #Here I am calculating the mean for embeedings
    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma
    
    #This function is for getting the required values for both refrence set and given set
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        try:
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        except ValueError:
            print("Sqrtm calculation produces singular product; adding epsilon to diagonal of cov estimates")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                print(f"Significant imaginary components found: {np.max(np.abs(covmean.imag))}")
                covmean = covmean.real + eps * np.eye(sigma1.shape[0])
                covmean = linalg.sqrtm(covmean.dot(covmean)).real
            else:
                covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    #This function is there for calling out mentioned function and providing with the result
    
    def score_from_audio(self, real_audio_list, generated_audio_list):
        try:
            embds_real = self.get_embeddings_from_audio(real_audio_list)
            embds_generated = self.get_embeddings_from_audio(generated_audio_list)
            
            if len(embds_real) == 0:
                print("[Frechet Audio Distance] real audio set is empty, exiting...")
                return MetricOutput(mean=-1)

            if len(embds_generated) == 0:
                print("[Frechet Audio Distance] generated audio set is empty, exiting...")
                return MetricOutput(mean=-1)

            mu_real, sigma_real = self.calculate_embd_statistics(embds_real)
            mu_generated, sigma_generated = self.calculate_embd_statistics(embds_generated)

            fad_score = self.calculate_frechet_distance(mu_real, sigma_real, mu_generated, sigma_generated)

            return MetricOutput(mean=fad_score)

        except Exception as e:
            print(f"[Frechet Audio Distance] exception thrown, {str(e)}")
            return MetricOutput(mean=-1)

#Here I am finding the files for refrence set
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

#This is the main function

def main(real_audio_dir,generated_audio_dir):

    real_audio_files, generated_audio_files = find_matching_real_audio(generated_audio_dir, real_audio_dir)
    if not real_audio_files or not generated_audio_files:
        print("No matching audio files found.")
        return

    fad = FrechetAudioDistance(verbose=True)
    result = fad.score_from_audio(real_audio_files, generated_audio_files)
    print(f"FAD Score: {result.mean}")


#This is the argument parser to parse the directory name
parser = argparse.ArgumentParser()
parser.add_argument('--real_audio_dir', type=str, required=True)
parser.add_argument('--gen_audio_dir', type=str, required=True)
args = parser.parse_args()

#Here I am calling the main function
main(args.real_audio_dir,args.gen_audio_dir)



