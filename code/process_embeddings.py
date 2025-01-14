
import numpy as np
import os
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import json
import pandas as pd
import argparse

# Load a slightly modified version of the Stable Diffusion pipeline.
# This allows us to extract text embeddings directly (without generating images).
# from model.custom_sd import StableDiffusionPipeline
# from model.custom_vd import TextToVideoSDPipeline
# from model.custom_ad import AudioLDMPipeline




sys.path.append("/home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/code/model")
from MotionDiffuse.trainers import DDPMTrainer
from os.path import join as pjoin
import sys
sys.path.append("/home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/code/model/MotionDiffuse")
import MotionDiffuse.utils.paramUtil as paramUtil
from torch.utils.data import DataLoader
from MotionDiffuse.utils.plot_script import *
from MotionDiffuse.utils.get_opt import get_opt
from MotionDiffuse.datasets1.dataset import Text2MotionDataset
import sys
sys.path.append("/home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/code/model/MotionDiffuse")
# from datasets.evaluator_models import MotionLenEstimatorBiGRU

from MotionDiffuse.trainers import DDPMTrainer
sys.path.append("/home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/code/model/MotionDiffuse")
from MotionDiffuse.models import MotionTransformer
from MotionDiffuse.utils.word_vectorizer import WordVectorizer, POS_enumerator
from MotionDiffuse.utils.utils import *
from MotionDiffuse.utils.motion_process import recover_from_ric



def save_to_path(emb, path):
    """Save embeddings to disk."""
    try:
        with open(path, 'wb') as wf:
            np.save(wf, emb)
    except:
        print("Error with", path)
    return path


if __name__ == '__main__':

    batch_size = 128

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    clip_output_dir = '/home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/data/T-X_pair_data/motion_t2m/embed'
    # synthesize_path = '../data/synthesize_data/synthesize_data.json'

    # video_path = '../data/T-X_pair_data/webvid/webvid.json'
    # audio_path = '../data/T-X_pair_data/audiocap/audiocap.json'
    # img_path = '../data/T-X_pair_data/cc3m/cc3m.json'

    # image_generation_ckpt_path = 'runwayml/stable-diffusion-v1-5'
    # video_generation_ckpt_path = 'cerspense/zeroscope_v2_576w'
    # audio_generation_ckpt_path = 'cvssp/audioldm-l-full'

    # data_path = sys.argv[1]
    # modality = sys.argv[2]
    # clip_output_dir = sys.argv[3]
    # ckpt_path = sys.argv[4]
    modality = 'motion'
    data_path = '/home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/data/T-X_pair_data/motion_t2m/motion.json'
    if not os.path.exists(clip_output_dir):
        os.makedirs(clip_output_dir, exist_ok=True)

    # Get existing files, so that we don't recompute them.
    existing_files = set([f.strip('.npy') for f in os.listdir(clip_output_dir)])

    caption_list = []
    name_list = []
    if modality == 'audio':
        print('extract audio caption embedding')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for row in tqdm(data, total=len(data)):
            one_audio_name, one_caption = row["audio_name"], row["caption"]
            if one_audio_name not in existing_files:
                caption_list.append(one_caption)
                name_list.append(one_audio_name)
        pipe = AudioLDMPipeline.from_pretrained(ckpt_path, torch_dtype=dtype)
        if not torch.cuda.is_available():
            print('WARNING: using CPU, this will be slow!')
        else:
            pipe = pipe.to("cuda")
    elif modality == 'image':
        print('extract image caption embedding')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for row in tqdm(data, total=len(data)):
            one_image_name, one_caption = row["image_name"], row["caption"]
            if one_image_name not in existing_files:
                caption_list.append(one_caption)
                name_list.append(one_image_name)
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=dtype)
        if not torch.cuda.is_available():
            print('WARNING: using CPU, this will be slow!')
        else:
            pipe = pipe.to("cuda")
    elif modality == 'video':
        print('extract video caption embedding')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for row in tqdm(data, total=len(data)):
            one_video_name, one_caption = row["video_name"], row["caption"]
            if one_video_name not in existing_files:
                caption_list.append(one_caption)
                name_list.append(one_video_name)
        pipe = TextToVideoSDPipeline.from_pretrained(ckpt_path, torch_dtype=dtype)
        if not torch.cuda.is_available():
            print('WARNING: using CPU, this will be slow!')
        else:
            pipe = pipe.to("cuda")
    elif modality == 'motion':
        print('extract video caption embedding')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for row in tqdm(data, total=len(data)):
            one_video_name, one_caption = row["file_name"], row["caption"]
            if one_video_name not in existing_files:
                caption_list.append(one_caption)
                name_list.append(one_video_name)




    opt_path = '/home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/code/model/MotionDiffuse/checkpoints/t2m/t2m_motiondiffuse/opt.txt'
    device = torch.device('cuda')
    opt = get_opt(opt_path,device= device)
    # max_motion_length = 196
    # num_layers = 8
    # no_clip = False
    # no_eff = False
    encoder = MotionTransformer(
        input_feats = 263,
        num_frames = opt.max_motion_length,
        num_layers = opt.num_layers,
        latent_dim= opt.latent_dim,
        no_clip= opt.no_clip,
        no_eff= opt.no_eff)
    
    opt.do_denoise = True

    # assert opt.dataset_name == "t2m"
    # assert self.args['motion_length'] <= 196
    opt.data_root = '/home/ltdoanh/jupyter/jupyter/ldtan/HumanML3D/HumanML3D'
    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.text_dir = pjoin(opt.data_root, 'texts')
    opt.joints_num = 22
    opt.dim_pose = 263
    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)
    num_classes = 200 // opt.unit_length
    train_split_file = pjoin(opt.data_root, 'train.txt')
    # mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    # std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    mean = np.load('/home/ltdoanh/jupyter/jupyter/ldtan/MotionDiffuse/checkpoints/t2m/t2m_motiondiffuse/meta/mean.npy')
    std = np.load('/home/ltdoanh/jupyter/jupyter/ldtan/MotionDiffuse/checkpoints/t2m/t2m_motiondiffuse/meta/std.npy')

    encoder = encoder.to(device)
    trainer = DDPMTrainer(opt, encoder)
    # trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
    trainer.load('/home/ltdoanh/jupyter/jupyter/ldtan/MotionDiffuse/checkpoints/t2m/t2m_motiondiffuse/model/latest.tar')

    trainer.eval_mode()
    trainer.to(device)
    opt.times = 50
    mean_data = np.load('/home/ltdoanh/jupyter/jupyter/ldtan/HumanML3D/HumanML3D/Mean.npy')
    std_data =  np.load('/home/ltdoanh/jupyter/jupyter/ldtan/HumanML3D/HumanML3D/Std.npy')
    motion_dataset = Text2MotionDataset(opt, mean_data, std_data, train_split_file, opt.times, False)
    # captions = []
    # caption, motion, m_length= motion_dataset.__getitem__(10)

    


    num_batches = int(np.ceil(len(caption_list) / batch_size))
    # input_text = [conversation for conversation in texts]
    
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_captions = caption_list[start_idx:end_idx]
        batch_ids = name_list[start_idx:end_idx]
        # prompt_embeds = pipe(batch_captions, return_prompts_only=True).detach().cpu().numpy()
        prompt_embeds = encoder.encode_text(batch_captions, device).detach().cpu().numpy()
        Parallel(n_jobs=8)(delayed(save_to_path)(
        prompt_embeds[j, :, ...], os.path.join(clip_output_dir, f'{batch_ids[j]}.npy')
    ) for j in range(prompt_embeds.shape[0]))
    print('Extract embeddings in batches.')
    # num_batches = int(np.ceil(len(caption_list) / batch_size))
    # for i in tqdm(range(num_batches)):
    #     start_idx = i * batch_size
    #     end_idx = start_idx + batch_size
    #     batch_captions = caption_list[start_idx:end_idx]
    #     batch_ids = name_list[start_idx:end_idx]
    #     prompt_embeds = pipe(batch_captions, return_prompts_only=True).detach().cpu().numpy()

    #     # Save embeddings to disk in parallel.
    #     Parallel(n_jobs=8)(delayed(save_to_path)(
    #         prompt_embeds[j, :, ...], os.path.join(clip_output_dir, f'{batch_ids[j]}.npy')
    #     ) for j in range(prompt_embeds.shape[0]))
