#Here I am importing all the essential libraries needed further
import time
import torchaudio
from audiocraft.models import MusicGen
from transformers import get_scheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
import wandb
from torch.utils.data import Dataset
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
import os
import numpy as np
import warnings 
import json

warnings.filterwarnings('ignore') 


#Here I am setting the seed function 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Here I am calling the set_seed function at the beginning 
set_seed(42)

#This class is for mapping the dataset so that it can load and use further
class Paired_Dataset(Dataset):
    def __init__(self, json_file):
        #Here I am calling the load json file function 
        self.data_map = self.load_data(json_file)
    
    #This defines the load .json file function  which have info about .txt file and .wav file
    def load_data(self, json_file):
        with open(json_file, 'r') as f:
            data_map = json.load(f)
        return data_map
    
    #This function is described here to return the length of data_dictionary
    def __len__(self):
        return len(self.data_map)

    #This function is all about getting the audio and label inside .json file
    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data["audio"]
        label = data.get("label", "")
        return audio, label


#This function is all about preprocessing the audio file here we are passing audio path, MusicGen Model, and duration for audio file which in our case is fixed for 30s chunks
def preprocess_audio(audio_path, model, duration: int = 30):
    try:
        #Here I am loading the audio sample
        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
        #Here I am taking the mean and changing it in one dimension
        wav = wav.mean(dim=0, keepdim=True)
        #Here I am making sure if the my length is according to required length otherwise it will get padded
        required_length = model.sample_rate * duration
        if wav.shape[1] < required_length:
            padding = required_length - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, padding))
        
        wav = wav.cuda().unsqueeze(1)
        
        #Here I am using the encodec model of MusicGen to compress and get audio embeedings
        with torch.no_grad():
            gen_audio = model.compression_model.encode(wav)

        codes, scale = gen_audio
        assert scale is None
        return codes
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        return None

#Here I am calculating the one hot encode value to further use for loss calculation
def calculate_one_hot_encode(audio, num_classes=2048):
    shape = audio.shape
    one_hot_encode = torch.zeros((shape[0], shape[1], num_classes))
    for i in range(shape[0]):
        for j in range(shape[1]):
            index = audio[i, j].item()
            one_hot_encode[i, j, index] = 1
    return one_hot_encode
    

#This is the main training and validating function
def train(
    train_dataset_path: str, # path for training json file
    valid_dataset_path: str, # path for validation json file
    lr: float, # given learning rate
    epochs: int, # number of epochs considered to run the loop
    using_wandb: bool,  # using weights and bias for logging
    save_file: str,  # name of model to be saved
    grad_acc: int, # grad accumulation
    weight_decay: float, # weight decay defined
    batch_size: int,  # batch size defined
    warmup_steps: int, # warmup stps defined
    use_cfg: bool, # For classifier free guidance
    patience: int = 7,  # Early stopping patience
    lr_reduce_patience: int = 5,  # Learning rate reduction patience
    lr_reduce_factor: float = 0.1  # Learning rate reduction factor
):
    # print(warmup_steps)
    if using_wandb:
        print("Logs are monitored using Wandb")
        run = wandb.init(project="audiocraft")
    
    #device for using cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #here I am loading the pretrained MusicGen-small model
    model = MusicGen.get_pretrained('small')
    
    #The language model is being used here
    model.lm = model.lm.to(torch.float32)  # important

    #This is for loading the training and preprocessing the training and validation dataset.
    train_dataset = Paired_Dataset(train_dataset_path)
    val_dataset = Paired_Dataset(valid_dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Started Fine Tuning model")

    
    # optimizer = AdamW(model.lm.parameters(),lr=lr,betas=(0.9, 0.95),weight_decay=weight_decay,)
    #Here I am loading the optimizer for model which is AdamW
    optimizer = AdamW(model.lm.parameters(),lr=lr,betas=(0.9, 0.99),weight_decay=weight_decay,)
    
    #Here I am loading the cosine scheduler for the model
    scheduler = get_scheduler("cosine",optimizer,warmup_steps,int(epochs * len(train_dataloader) / grad_acc))
    #Here I am initiliazing the loss as cross entropy loss
    criterion = nn.CrossEntropyLoss()

    #Here I am talking about saving path for final model and create a directory if it doesn't exist
    save_path = "models/"
    os.makedirs(save_path, exist_ok=True)

    #Here I am intilializing values to get a knowldege about the step
    curr_step = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    epochs_no_reduce = 0

    #Now we are entering into epoch loop
    for epoch in range(epochs):
        #Here I am initializing epoch start time
        ep_start = time.time()
        epoch_loss=0.0
        num_batches=0
        #Here The model goes into training mode
        model.lm.train()
        
        #Here I am iterating the training data
        for batch_idx, (audio, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            all_codes = []
            all_texts = []
            
            #Now we are accessing the audio and label files
            for inner_audio, inner_label in zip(audio, label):
                inner_audio = preprocess_audio(inner_audio, model)
                
                #If we are using classifier free guidance then we need to increase the size of it so that it can apply
                if use_cfg:
                    codes = torch.cat([inner_audio, inner_audio], dim=0)
                else:
                    codes = inner_audio
                
                #Now we are appending the details of code which is embeeded music in our case
                all_codes.append(codes)
                
                #Similarly we are reading the label which is description and storing it in variable
                all_texts.append(open(inner_label, "r").read().strip())
            
            # print(all_codes)
            
            #Here I am calling the function which will prepare conditioning attributes for our generational model here you can give melody and all but I am only providing the descriptions
            attributes, _ = model._prepare_tokens_and_attributes(all_texts, None)
            conditions = attributes
            
            #If you are using classifier free guidanve then It will drop some conditions and append them together
            if use_cfg:
                null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                conditions = conditions + null_conditions
            
            #Now here I am converting them into tokens with the help of T5 tokenizer
            tokenized = model.lm.condition_provider.tokenize(conditions)
            
            # Here I am encoding conditions and fuse, both have a streaming cache to not recompute when generating.
            cfg_conditions = model.lm.condition_provider(tokenized)
            condition_tensors = cfg_conditions
            
            
            codes = torch.cat(all_codes, dim=0)
            # print(prompt)
            # print(conditions)
            # print(condition_tensors)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                
                #Here I am calling the predictions function and provided the condition tensore obtained from description, there is no need to give both conditions and condition tensors.
                model_output = model.lm.compute_predictions(codes=codes, conditions=[], condition_tensors=condition_tensors)
                
                codes = codes[0]
                
                #This is the output we are getting in the form of logits and movig it to cuda
                logits = model_output.logits[0].cuda()
                
                #We are also getting output as mask so we moved it to cuda as well for further calculation
                mask = model_output.mask[0].cuda().view(-1)
                
                #Now on the original codes we are converting it woth the help of one_hot_encode
                codes = calculate_one_hot_encode(codes, num_classes=2048).cuda()
                
                #Here we are applying masking logic to both original and predicted value
                masked_logits = logits.view(-1, 2048)[mask]
                masked_codes = codes.view(-1, 2048)[mask]
                
                #Here I am calculating the cross entropy loss using both values
                loss = criterion(masked_logits, masked_codes)
            
            curr_step += 1 / grad_acc
            loss.backward()

            #Here I am calculating the toal norm thing
            total_norm = 0
            for p in model.lm.condition_provider.parameters():
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except AttributeError:
                    pass
            total_norm = total_norm ** (1.0 / 2)
            epoch_loss += loss.item()
            num_batches+=1

            if batch_idx % grad_acc != grad_acc - 1:
                continue
            
            #Now all these are steps needed for gradient clipping and optimizer as well as scheduler
            torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 0.5)

            optimizer.step()
            scheduler.step(loss)

        
        #Here I am calculating losses per epoch and logging it using wandb
        epoch_loss /= num_batches
        if using_wandb:
                run.log(
                    {
                        "loss": epoch_loss,
                    }
                )
        
        #This is for printing the details
        print(f"Epoch {epoch+1} of {epochs} completed in {time.time() - ep_start:.2f}s Loss:{epoch_loss}")
        
      
        # Now we are starting with Validation loop
        
        #Here the model goes in evaluation mode
        model.lm.eval()
        val_loss = 0
        with torch.no_grad():
             #Now we are accessing the audio and label files
            for batch_idx, (audio, label) in enumerate(val_dataloader):
                all_codes = []
                all_texts = []
                
                 #Now we are accessing the audio and label files
                for inner_audio, inner_label in zip(audio, label):
                    inner_audio = preprocess_audio(inner_audio, model)
                    
                    #If we are using classifier free guidance then we need to increase the size of it so that it can apply
                    if use_cfg:
                        codes = torch.cat([inner_audio, inner_audio], dim=0)
                    else:
                        codes = inner_audio

                    all_codes.append(codes)
                    all_texts.append(open(inner_label, "r").read().strip())
                
                 #Here I am calling the function which will prepare conditioning attributes for our generational model here you can give melody and all but I am only providing the descriptions
                attributes, _ = model._prepare_tokens_and_attributes(all_texts, None)
                conditions = attributes
                if use_cfg:
                    null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
                    conditions = conditions + null_conditions
                    
                #Now here I am converting them into tokens with the help of T5 tokenizer
                tokenized = model.lm.condition_provider.tokenize(conditions)
                cfg_conditions = model.lm.condition_provider(tokenized)
                condition_tensors = cfg_conditions

                codes = torch.cat(all_codes, dim=0)
                
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model_output = model.lm.compute_predictions(
                        codes=codes, conditions=[], condition_tensors=condition_tensors
                    )

                    codes = codes[0]
                    logits = model_output.logits[0].cuda()
                    mask = model_output.mask[0].cuda().view(-1)

                    codes = calculate_one_hot_encode(codes, num_classes=2048).cuda().view(-1)

                    masked_logits = logits.view(-1, 2048)[mask]
                    masked_codes = codes.view(-1, 2048)[mask]

                    loss = criterion(masked_logits, masked_codes)
                    val_loss += loss.item()

        
        #Here I am calculating the validation loss and printing it

        val_loss /= len(val_dataloader)
        print(f"Validation Loss after Epoch {epoch+1}: {val_loss}")
        print()

        if using_wandb:
            run.log({"val_loss": val_loss})
        
        
        # Now we are checking for validation loss improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            #Here I am saving the best so far model
            torch.save(model.lm.state_dict(), f"{save_path}/{save_file}_best.pt")
        else:
            epochs_no_improve += 1
            epochs_no_reduce += 1

        # This block is intend for Early stopping
        if epochs_no_improve >= patience:
            print("Early stopping due to no improvement in validation loss.")
            break

        # Reduce learning rate if there is no improvement
        if epochs_no_reduce >= lr_reduce_patience:
            lr *= lr_reduce_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Reduced learning rate to {lr}")
            epochs_no_reduce = 0

    # Now we are saving the final model
    torch.save(model.lm.state_dict(), f"{save_path}/{save_file}_final.pt")
