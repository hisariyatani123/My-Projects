#Here I am calling all the necessary modules
import time
import torchaudio
from audiocraft.models import MusicGen
from transformers import get_scheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import wandb
from torch.utils.data import Dataset
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
import os
import random
import numpy as np
import json
import warnings

warnings.filterwarnings('ignore')

#This is for setting the seed function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call the set_seed function at the beginning of your script
set_seed(42)


#This is for getting data from defined json file
class AudioDataset(Dataset):
    def __init__(self, json_file):
        self.data_map = self.load_data(json_file)

    def load_data(self, json_file):
        with open(json_file, 'r') as f:
            data_map = json.load(f)
        return data_map

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data["audio"]
        label = data.get("label", "")
        return audio, label

#This is for preprocessing the audio
def preprocess_audio(audio_path, model: MusicGen, duration: int = 30):
    try:
        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
        wav = wav.mean(dim=0, keepdim=True)
        required_length = model.sample_rate * duration
        if wav.shape[1] < required_length:
            padding = required_length - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, padding))
        
        wav = wav.cuda()
        wav = wav.unsqueeze(1)

        with torch.no_grad():
            gen_audio = model.compression_model.encode(wav)

        codes, scale = gen_audio
        assert scale is None

        return codes
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return None

#This is for one hot encode calcualtion
def one_hot_encode(tensor, num_classes=2048):
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], num_classes))
    for i in range(shape[0]):
        for j in range(shape[1]):
            index = tensor[i, j].item()
            one_hot[i, j, index] = 1
    return one_hot

#This function is for preparing the tokens and attributes for textual prompt
def prepare_conditions(texts, model):
    attributes, _ = model._prepare_tokens_and_attributes(texts, None)
    return attributes

#This function is for streamlined preprocessing step
def preprocess_dataset(dataloader, model, use_cfg, output_dir, name):
    all_codes_list = []
    all_condition_tensors_dict =[]

    for batch_idx, (audio, label) in enumerate(dataloader):
        all_codes, texts = [], []
        for inner_audio, l in zip(audio, label):
            inner_audio = preprocess_audio(inner_audio, model)
            if inner_audio is None:
                continue
            if use_cfg:
                codes = torch.cat([inner_audio, inner_audio], dim=0)
            else:
                codes = inner_audio
            
            all_codes.append(codes)
            texts.append(open(l, "r").read().strip())

        if not all_codes:
            continue

        conditions = prepare_conditions(texts, model)
        if use_cfg:
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                conditions = conditions + null_conditions

        tokenized = model.lm.condition_provider.tokenize(conditions)
        cfg_conditions = model.lm.condition_provider(tokenized)
        condition_tensors = cfg_conditions
        
        codes = torch.cat(all_codes, dim=0)

        all_codes_list.append(codes)
        all_condition_tensors_dict.append(condition_tensors)

    torch.save(all_codes_list, os.path.join(output_dir, f"{name}_codes.pt"))
    torch.save(all_condition_tensors_dict, os.path.join(output_dir, f"{name}_condition_tensors.pt"))

#This class is for loading the precomputed dataset
class PrecomputedDataset(Dataset):
    def __init__(self, codes_path, condition_tensors_path):
        self.codes = torch.load(codes_path)
        self.condition_tensors = torch.load(condition_tensors_path)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        condition_tensor = self.condition_tensors[idx]
        return code, condition_tensor

def train(
    train_dataset_path: str,
    valid_dataset_path: str,
    precompute_dir: str,
    lr: float,
    epochs: int,
    using_wandb: bool,
    save_file:str,
    grad_acc: int,
    weight_decay: float,
    warmup_steps: int,
    batch_size: int,
    use_cfg: bool,
    patience: int = 5,  # Early stopping patience
    lr_reduce_patience: int = 3,  # Learning rate reduction patience
    lr_reduce_factor: float = 0.1  # Learning rate reduction factor
):
    if using_wandb:
        print("Using Wandb")
        run = wandb.init(project="audiocraft")
    
    #This is the model loading step
    model = MusicGen.get_pretrained('small')
    model.lm = model.lm.to(torch.float32)  # important
    
    #This is for loading the data step
    train_dataset = AudioDataset(train_dataset_path)
    val_dataset = AudioDataset(valid_dataset_path)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    #This is for preprocessing the data step
    preprocess_dataset(train_dataloader, model, use_cfg, precompute_dir,'train')
    preprocess_dataset(val_dataloader, model, use_cfg,precompute_dir,'test') 
    
    train_codes = torch.load(os.path.join(precompute_dir, 'train_codes.pt'))
    train_condition_tensors = torch.load(os.path.join(precompute_dir, 'train_condition_tensors.pt'))
    val_codes = torch.load(os.path.join(precompute_dir, 'test_codes.pt'))
    val_condition_tensors = torch.load(os.path.join(precompute_dir, 'test_condition_tensors.pt'))
    
    learning_rate = lr
    model.lm.train()

    scaler = torch.cuda.amp.GradScaler()

    print("Started tuning")

    optimizer = AdamW(model.lm.parameters(),lr=learning_rate,betas=(0.9, 0.99),weight_decay=weight_decay,)
    scheduler = get_scheduler("cosine",optimizer,warmup_steps,int(epochs * len(train_codes) / grad_acc),)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = epochs

    save_path = "models/"
    os.makedirs(save_path, exist_ok=True)

    current_step = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    epochs_no_reduce = 0
    
    #Here the loop starts and training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        model.lm.train()

        for batch_idx in range(0, len(train_codes)):
            optimizer.zero_grad()
            codes_batch=train_codes[batch_idx]
            condition_tensors_batch=train_condition_tensors[batch_idx]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lm_output = model.lm.compute_predictions(
                    codes=codes_batch, 
                    conditions=[], 
                    condition_tensors=condition_tensors_batch
                )

                codes=codes_batch[0]
                logits = lm_output.logits[0]
                mask = lm_output.mask[0]

                codes = one_hot_encode(codes, num_classes=2048)

                codes = codes.cuda()
                logits = logits.cuda()
                mask = mask.cuda()

                mask = mask.view(-1)
                masked_logits = logits.view(-1, 2048)[mask]
                masked_codes = codes.view(-1, 2048)[mask]

                loss = criterion(masked_logits, masked_codes)
            
            batch_idx+=1
            current_step += 1 / grad_acc
            loss.backward()

            total_norm = 0
            for p in model.lm.parameters():
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except AttributeError:
                    pass
            total_norm = total_norm ** (1.0 / 2)

            epoch_loss += loss.item()
            num_batches += 1

            if current_step % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 0.5)

                optimizer.step()
                scheduler.step(loss)

        epoch_loss /= num_batches
        print(f"Epoch {epoch+1} of {num_epochs} completed in {time.time() - epoch_start_time:.2f}s Loss: {epoch_loss}")
        if using_wandb:
            run.log({"train_loss": epoch_loss})
            
        # This loop is for validation loop
        model.lm.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx in range(0, len(val_codes)):
                
                val_codes_batch=val_codes[batch_idx]
                val_condition_tensors_batch=val_condition_tensors[batch_idx]

                lm_output = model.lm.compute_predictions(
                    codes=val_codes_batch, 
                    conditions=[], 
                    condition_tensors=val_condition_tensors_batch
                )

                codes = val_codes_batch[0]
                logits = lm_output.logits[0]
                mask = lm_output.mask[0]

                codes = one_hot_encode(codes, num_classes=2048)

                codes = codes.cuda()
                logits = logits.cuda()
                mask = mask.cuda()

                mask = mask.view(-1)
                masked_logits = logits.view(-1, 2048)[mask]
                masked_codes = codes.view(-1, 2048)[mask]

                loss = criterion(masked_logits, masked_codes)
                val_loss += loss.item()
                    
                batch_idx+=1
                
        val_loss /= (len(val_codes))
        print(f"Validation Loss after Epoch {epoch+1}: {val_loss}")

        if using_wandb:
            run.log({"val_loss": val_loss})

        # This is for checking for validation loss improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.lm.state_dict(), f"{save_path}/{save_file}_best.pt")
        else:
            epochs_no_improve += 1
            epochs_no_reduce += 1

        # This is for early stopping
        if epochs_no_improve >= patience:
            print("Stopping early due to no improvement in validation loss.")
            break

        # This is for reducing learning rate if there is no improvement
        if epochs_no_reduce >= lr_reduce_patience:
            learning_rate *= lr_reduce_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            print(f"Reduced learning rate to {learning_rate}")
            epochs_no_reduce = 0

    # Here I am saving the final model
    torch.save(model.lm.state_dict(), f"{save_path}/{save_file}_final.pt")