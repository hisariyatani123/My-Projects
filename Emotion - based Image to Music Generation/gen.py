#Here I am importing all the necessary libraries
import torchaudio
from tqdm import trange
from audiocraft.models import MusicGen
import torch
import argparse
import warnings

warnings.filterwarnings('ignore')

#Here I am calling the argument parse to parse the passed arguments value
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, required=True) # this is basically the prompt 
parser.add_argument('--weights_path', type=str, required=False, default=None) # This is the path for saved weight of model
parser.add_argument('--save_path', type=str, required=False, default='test.wav') # This is the path for saving the generated music
parser.add_argument('--duration', type=float, required=False, default=30) # This is the durayion of generated music
parser.add_argument('--use_sampling', type=bool, required=False, default=1) # These all are defaults values for generating the music based on sequences like top_k for selcting topmost values, temperature to cpntrol creatibity and others
parser.add_argument('--two_step_cfg', type=bool, required=False, default=0)
parser.add_argument('--top_k', type=int, required=False, default=250)
parser.add_argument('--top_p', type=float, required=False, default=0.0)
parser.add_argument('--temperature', type=float, required=False, default=1.0)
parser.add_argument('--cfg_coef', type=float, required=False, default=3.0)
args = parser.parse_args()

#Here I am loading the pretrained small model
model = MusicGen.get_pretrained('small')

self = model

#Here I am loading the weights of saved model
if args.weights_path is not None:
    self.lm.load_state_dict(torch.load(args.weights_path))
    
#Here I am preparing the attributes for given descriptions
attributes, prompt_tokens = self._prepare_tokens_and_attributes([args.prompt], None)
# print("attributes:", attributes)
# print("prompt_tokens:", prompt_tokens)

#This is for calculating the maximum length of generated output based on required music length
duration = args.duration
max_len= int(duration * (self.frame_rate))

#Here I am setting all the parameters required for generator function
self.generation_params = {
    'max_gen_len':max_len,
    'use_sampling': args.use_sampling,
    'temp': args.temperature,
    'top_k': args.top_k,
    'top_p': args.top_p,
    'cfg_coef': args.cfg_coef,
    'two_step_cfg': args.two_step_cfg,
}

with self.autocast:
    #Here I am calling the generate function based on attributes and parameters
    gen_tokens = self.lm.generate(prompt_tokens, attributes, callback=None, **self.generation_params)

#Here I am calling the encodec decoder function to decode it and produce the result
assert gen_tokens.dim() == 3
with torch.no_grad():
    gen_audio = self.compression_model.decode(gen_tokens, None)
    
# print("gen_audio information")
# print("Shape:", gen_audio.shape)
# print("Dtype:", gen_audio.dtype)
# print("Contents:", gen_audio)


#Here I am moving the generated audio tensor to cpu and saving it at define location
gen_audio = gen_audio.cpu()
torchaudio.save(args.save_path, gen_audio[0], self.sample_rate)
print("Saved the file")