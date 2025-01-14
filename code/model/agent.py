import os.path

import torch
from header import *
import os
import torch.optim as optim
# from model.MotionDiffuse.datasets1 import Text2MotionDataset
# from model.MotionDiffuse.datasets1 import build_dataloader
import sys
from os.path import join as pjoin
sys.path.append('/home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/code/model/MotionDiffuse/models')
import model.MotionDiffuse.utils.paramUtil as paramUtil
from model.MotionDiffuse.options.train_options import TrainCompOptions
from model.MotionDiffuse.utils.plot_script import *
# from model.MotionDiffuse.models import MotionTransformer
# from model.MotionDiffuse.trainers import DDPMTrainer  
from model.MotionDiffuse.utils.utils import print_current_loss
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
def build_models(opt, dim_pose):
    encoder = MotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder

class WarmupThenDecayScheduler:
    def __init__(self, optimizer, warmup_min_lr, warmup_max_lr, warmup_num_steps, total_num_steps, step_size, gamma):
        self.optimizer = optimizer
        self.warmup_min_lr = warmup_min_lr
        self.warmup_max_lr = warmup_max_lr
        self.warmup_num_steps = warmup_num_steps
        self.total_num_steps = total_num_steps
        self.step_size = step_size
        self.gamma = gamma
        self.current_step = 0

    def get_lr(self):
        if self.current_step < self.warmup_num_steps:
            # Linear warmup
            lr = self.warmup_min_lr + (self.warmup_max_lr - self.warmup_min_lr) * (self.current_step / self.warmup_num_steps)
        else:
            # Exponential decay
            decay_steps = (self.current_step - self.warmup_num_steps) // self.step_size
            lr = self.warmup_max_lr * (self.gamma ** decay_steps)
        return lr

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Usage exampl
def build_models(opt, dim_pose):
    encoder = MotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder

def initialize_new_parameters(model, checkpoint_state):
    model_state = model.state_dict()  # Get the current model's state dictionary

    for name, param in model.named_parameters():
        if name not in checkpoint_state:
            print(name)  # Only initialize new parameters
            if param.requires_grad:  # Ensure it's a trainable parameter
                if param.dim() > 1:  # For weights (multidimensional)
                    nn.init.kaiming_normal_(param)
                else:  # For biases (1D tensors)
                    nn.init.zeros_(param)


class DeepSpeedAgent:

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_parameters(self.args['save_path'], self.args['stage'])

        self.print_model_parameters()
        self.writer = SummaryWriter(args['log_path'])

       
        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(
            self.args['total_steps'] * self.args['warmup_rate']))
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # self.ds_engine = self.model
        # os.environ['RANK'] = '0'
        # os.environ['WORLD_SIZE'] = '1'
        # self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
        #     model=self.model,
        #     model_parameters=self.model.parameters(),
        #     config_params=ds_params,
        #     dist_init_required=False,
        #     args=types.SimpleNamespace(**args)
        # )
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0001,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.001
        )
        

        self.scheduler = WarmupThenDecayScheduler(
            self.optimizer,
            warmup_min_lr=0,
            warmup_max_lr=0.0005,
            warmup_num_steps=10,
            total_num_steps=10000,
            step_size=1000,
            gamma=0.95
        )
        parser = TrainCompOptions()
        opt = parser.parse()

        opt.device = torch.device("cuda")
        torch.autograd.set_detect_anomaly(True)

        opt.save_root = '/home/ltdoanh/jupyter/jupyter/ldtan/MotionDiffuse/t2m/t2m_new_ver4'
        opt.model_dir = pjoin(opt.save_root, 'model')
        opt.meta_dir = pjoin(opt.save_root, 'meta')

    

        opt.dataset_name == 't2m'
        opt.data_root = '/home/ltdoanh/jupyter/jupyter/ldtan/HumanML3D/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        radius = 240 * 8
        fps = 12.5
        dim_pose = 263
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.t2m_kinematic_chain



        dim_word = 300
        mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        std = np.load(pjoin(opt.data_root, 'Std.npy'))

        train_split_file = pjoin(opt.data_root, 'train.txt')
        encoder = build_models(opt, dim_pose)
    
        self.encoder = encoder.cuda()
        
        
        # Load checkpoint
        checkpoint = torch.load('/home/ltdoanh/jupyter/jupyter/ldtan/MotionDiffuse/t2m/t2m_new_ver4/model/fushion_checkpoint_bs_8.tar', map_location='cuda')

        # Load pretrained weights for the encoder (matching keys only)
        checkpoint_state = checkpoint['encoder']
        model_state = self.encoder.state_dict()

        # Load only matching keys from the checkpoint
        pretrained_dict = {k: v for k, v in checkpoint_state.items() if k in model_state and model_state[k].shape == v.shape}
        self.encoder.load_state_dict(pretrained_dict, strict=False)

        # Initialize new layers in the encoder that were not in the checkpoint
        initialize_new_parameters(self.encoder, checkpoint_state)

        # Log which layers were loaded or initialized
        for name, param in self.encoder.named_parameters():
            if name in pretrained_dict:
                print(f"Loaded pretrained layer: {name}")
            else:
                print(f"Initialized new layer: {name}")

        # Create a new optimizer for the encoder (fresh start)
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=2e-4)
        for group in self.opt_encoder.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    print(f"Parameter: {param.shape}, Gradient: {param.grad.shape}")
                else:
                    print(f"Parameter: {param.shape}, Gradient: None")

        # Add new parameters to the optimizer if they were not in the checkpoint (avoid duplicates)
        existing_params = set(p for group in self.opt_encoder.param_groups for p in group['params'])
        new_params = [p for name, p in self.encoder.named_parameters() if p.requires_grad and p not in existing_params]

        if new_params:
            self.opt_encoder.add_param_group({'params': new_params})
            print(f"Added {len(new_params)} new parameters to the optimizer.")


        self.trainer = DDPMTrainer(opt, self.encoder) 
        self.encoder = self.encoder.cuda()
        for param in self.encoder.parameters():
            param.requires_grad = True

        # self.trainer = DDPMTrainer(opt, self.encoder)
        self.mse_criterion = torch.nn.MSELoss(reduction='none')    
        @torch.no_grad()
        def predict(self):
            self.ds_engine.module.eval()
            output = self.ds_engine.generate(self.args)
            return output
        
    # def train_model(self, batch, current_step=0, pbar=None):
    #     self.ds_engine.module.train()

    #     loss, mle_acc, mse_loss = self.ds_engine(batch)
    #     # loss, mle_acc, mse_loss = self.model(batch)
    #     self.writer.add_scalar('loss', loss, current_step)
    #     self.writer.add_scalar('mle_acc', mle_acc, current_step)
    #     # if isinstance(mse_loss, list):
    #     #     self.writer.add_scalar('img_mse_loss', mse_loss[0], current_step)
    #     #     self.writer.add_scalar('vid_mse_loss', mse_loss[1], current_step)
    #     #     self.writer.add_scalar('aud_mse_loss', mse_loss[2], current_step)
    #     if isinstance(mse_loss, torch.Tensor):
    #         self.writer.add_scalar('mse_loss', mse_loss, current_step)
    #     else:
    #         pass
    #     # self.writer.add_scalar('mse_loss', mse_loss, current_step)

    #     self.ds_engine.backward(loss)
    #     self.ds_engine.step()
    #     # pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}; mse_loss: {round(mse_loss[0].item(), 4)} ')
    #     pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
    #     pbar.update(1)
    #     if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
    #         elapsed = pbar.format_dict['elapsed']
    #         rate = pbar.format_dict['rate']
    #         remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
    #         remaining = str(datetime.timedelta(seconds=remaining))
    #         logging.info(
    #             f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
    #         # ; mse_loss: {round(mse_loss[0].item(), 4)}
    #     mle_acc *= 100
    #     return mle_acc
    # def train_model(self, batch, motions, m_lens,start_time, it, i, epoch,opt_encoder,encoder,trainer,current_step=0, pbar=None, a=0,):

    def train_model(self, batch, motions, m_lens, i, start_time, it, epoch, current_step=0, pbar=None):
        self.model.train()
        embeddings, texts = self.model(batch) # embedding size (1, 77, 512)
        text_promts = [f'generate for me a video that {desc}' for desc in list(texts)]
        text_promts_embed, text_promts_out = self.encoder.encode_promt(text_promts, self.device)  #[bs, 2048]
        logs = OrderedDict()
        motions = motions.to(self.device).float()
        x_start = motions
        B, T = x_start.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
        t, _ = self.trainer.sampler.sample(B, x_start.device)
        text_promts_embed = text_promts_embed.to(self.device)
        text_promts_out = text_promts_out.to(self.device)
        embeddings = embeddings.to(self.device)
        output = self.trainer.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            text_promts_embed = text_promts_embed,
            text_promts_out =text_promts_out,
            model_kwargs={"text": embeddings, "length": cur_len}
        )
        real_noise = output['target']
        fake_noise = output['pred']
        real_noise = real_noise.to(self.device)
        fake_noise = fake_noise.to(self.device)

        try:
            self.src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(self.device)
        except:
            self.src_mask = self.encoder.generate_src_mask(T, cur_len).to(self.device)
        self.optimizer.zero_grad()
        # self.trainer.zero_grad([self.opt_encoder])
        self.opt_encoder.zero_grad()
        loss_logs, loss_mot_rec= self.trainer.backward_G(real_noise, fake_noise, self.src_mask)
        loss_mot_rec = loss_mot_rec + self.encoder.get_out_Loss() + self.encoder.get_proj_Loss()
        loss_mot_rec = torch.where(torch.isnan(loss_mot_rec), torch.tensor(0.0213), loss_mot_rec)
        loss_mot_rec = loss_mot_rec.to(self.device)
        loss_mot_rec = loss_mot_rec.detach().requires_grad_()


        # cosine_loss = self.encoder.get_Cosine_Similarity_Loss()
        # l2_loss = self.encoder.get_out_Loss() + self.encoder.get_proj_Loss()
        with open('/home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/code/loss/output.txt', 'a') as f:
            # print("epoch: " + str(epoch)+ " i: " + str(i) + " Cosine_loss is: "+ str(cosine_loss), file = f)
            print("epoch: " + str(epoch)+ " i: " + str(i) + " loss motion diffuse: " + str (loss_mot_rec), file = f)
            print("epoch: " + str(epoch)+ " i: " + str(i) + " l2_loss_out: " + str (self.encoder.get_out_Loss()), file = f)
            print("epoch: " + str(epoch)+ " i: " + str(i) + " l2_loss_proj: " + str (self.encoder.get_proj_Loss()), file = f)
        print(loss_mot_rec)
        # loss_mot_rec.backward()
        
        try:
            loss_mot_rec.backward()
        except:
            print("nan losssc")
            
        
        

        # self.trainer.clip_norm([self.encoder])
        clip_grad_norm_(self.encoder.parameters(), 0.2)
        clip_grad_norm_(self.model.parameters(), 0.2)
        # self.trainer.step([self.opt_encoder])
        self.opt_encoder.step()
        self.optimizer.step()
        self.scheduler.step()
        log_dict = loss_logs
        for k, v in log_dict.items():
            if k not in logs:
                logs[k] = v
            else:
                logs[k] += v
        mean_loss = OrderedDict({})
        for tag, value in logs.items():
            mean_loss[tag] = value / 50
        logs = OrderedDict()
        print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)
        it += 1
        if it % 50 == 0:
            mean_loss = OrderedDict({})
            for tag, value in logs.items():
                mean_loss[tag] = value / 50
            logs = OrderedDict()
            print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)
        if i%500 ==0:
            self.trainer.save("/home/ltdoanh/jupyter/jupyter/ldtan/MotionDiffuse/t2m/t2m_new_ver4/model/fushion_checkpoint_bs_8.tar",epoch,it,self.opt_encoder)
        #     self.trainer.save("/home/ltdoanh/jupyter/jupyter/ldtan/MotionDiffuse/t2m/t2m_new_ver4/model/fushion_checkpoint_bs_32.tar",epoch,it,self.opt_encoder)
            
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.scheduler.step()
        # loss, mle_acc, mse_loss = self.model(batch)
        # # Logging
        # self.writer.add_scalar('loss', loss.item(), current_step)
        # # Assuming mle_acc and mse_loss can be derived from the outputs
        # # mle_acc = self.compute_mle_acc(outputs, targets)
        # # mse_loss = self.compute_mse_loss(outputs, targets)
        # self.writer.add_scalar('mle_acc', mle_acc, current_step)
        # if isinstance(mse_loss, torch.Tensor):
        #     self.writer.add_scalar('mse_loss', mse_loss.item(), current_step)
        # else:
        #     pass
        # # Backward pass
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.scheduler.step()

        # # Logging progress
        # pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
        # pbar.update(1)
        # if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
        #     elapsed = pbar.format_dict['elapsed']
        #     rate = pbar.format_dict['rate']
        #     remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
        #     remaining = str(datetime.timedelta(seconds=remaining))
        #     logging.info(
        #         f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}'
        #     )
        #     with open('gpt_output_new.txt', 'a') as f:
        #         f.write(f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}' + '\n')

        # mle_acc *= 100
        # return mle_acc

    # def save_model(self, path, current_step):
    #     """
    #         this function also save the trainable parameters and specific name parameters
    #     """
    #     param_grad_dic = {
    #         k: v.requires_grad for (k, v) in self.ds_engine.module.named_parameters()
    #     }
    #     state_dict = self.ds_engine.module.state_dict()
    #     checkpoint = OrderedDict()
    #     for k, v in self.ds_engine.module.named_parameters():
    #         if v.requires_grad:
    #             checkpoint[k] = v
    #         if 'gen_text_hidden_fcs' in k:
    #             checkpoint[k] = v
    #         if 'gen_text_hidden_fcs_video' in k:
    #             checkpoint[k] = v
    #         if 'gen_text_hidden_fcs_audio' in k:
    #             checkpoint[k] = v
    #         if 'llama_proj' in k:
    #             checkpoint[k] = v
    #     torch.save(checkpoint, f'{path}/pytorch_model.pt')
    #     # save tokenizer
    #     self.model.llama_tokenizer.save_pretrained(path)
    #     # save configuration
    #     self.model.llama_model.config.save_pretrained(path)
    #     print(f'[!] save model into {path}')
    def save_model(self, path, current_step):
        """
            Lưu các tham số có thể huấn luyện và các tham số cụ thể theo tên.
        """
        # Lưu tham số có thể huấn luyện
        checkpoint = OrderedDict()
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                checkpoint[k] = v
            if 'gen_text_hidden_fcs' in k:
                checkpoint[k] = v
            if 'gen_text_hidden_fcs_video' in k:
                checkpoint[k] = v
            if 'gen_text_hidden_fcs_audio' in k:
                checkpoint[k] = v
            if 'gen_text_hidden_fcs_motion' in k:
                checkpoint[k] = v
            if 'llama_proj' in k:
                checkpoint[k] = v

        torch.save(checkpoint, f'{path}/pytorch_model.pt')

        # Lưu tokenizer
        self.model.llama_tokenizer.save_pretrained(path)

        # Lưu cấu hình
        self.model.llama_model.config.save_pretrained(path)

        print(f'[!] Model saved to {path}')

    def print_model_parameters(self, use_4bit=False):
        """
            Prints the number of trainable parameters in the model.
            """
        trainable_params = 0
        all_param = 0
        lora = 0
        image = 0
        video = 0
        audio = 0
        linear = 0
        llama = 0
        imagebind = 0
        motion =0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if 'lora' in name:
                lora += num_params
            elif 'gen_text_hidden_fcs_video' in name:
                video += num_params
            elif 'gen_text_hidden_fcs_audio' in name:
                audio += num_params
            elif 'gen_text_hidden_fcs_motion' in name:
                motion += num_params
            elif 'gen_text_hidden_fcs' in name:
                image += num_params
            elif 'llama_proj' in name:
                linear += num_params
            elif 'llama_model' in name:
                llama += num_params
            elif 'visual_encoder' in name:
                imagebind += num_params
            else:
                pass

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        if use_4bit:
            trainable_params /= 2
        print(
            f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
        )
        print(f'lora params: {lora:,d} || video params: {video:,d} || audio params: {audio:,d} || image params: {image:,d} || motion params: {motion:,d} ')
        print(f'linear params: {linear:,d} || imagebind params: {imagebind:,d} || llama params: {llama:,d}')

    # def load_parameters(self, path, stage=3):
    #     if os.path.exists(os.path.join(path, 'pytorch_model.pt')):
    #         print('loading parameters from {}'.format(self.args['save_path']))
    #         delta_ckpt = torch.load(f'{path}/pytorch_model.pt', map_location=torch.device('cuda'))
    #         checkpoint = OrderedDict()
    #         if stage == 3:
    #             for k, v in delta_ckpt.items():
    #                 if 'llama_model.model.embed_tokens.weight' in k:
    #                     checkpoint['llama_model.base_model.model.model.embed_tokens.weight'] = v
    #                 elif 'llama_model.lm_head.weight' in k:
    #                     checkpoint['llama_model.base_model.model.lm_head.weight'] = v
    #                 else:
    #                     checkpoint[k] = v
    #         else:
    #             checkpoint = delta_ckpt
    #         self.model.load_state_dict(checkpoint, strict=False)
    def load_parameters(self, path, stage=2):
        if os.path.exists('/home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/ckpt/delta_ckpt/nextgpt/7b_tiva_v0/pytorch_model.pt'):
            print('loading parameters from: /home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/ckpt/delta_ckpt/nextgpt/7b_tiva_v0/pytorch_model.pt')
            delta_ckpt = torch.load('/home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/ckpt/delta_ckpt/nextgpt/7b_tiva_v0/pytorch_model.pt', map_location=torch.device('cuda'))

            self.model.load_state_dict(delta_ckpt, strict=False)

            delta_ckpt_new = torch.load('/home/ltdoanh/jupyter/jupyter/ldtan/NExT-GPT/ckpt/test_8/pytorch_model.pt', map_location=torch.device('cuda'))
            for name, param in self.model.named_parameters():
                if 'gen_text_hidden_fcs_motion' in name:
                    param.data.copy_(delta_ckpt_new[name].data)
            checkpoint = OrderedDict()
            if stage == 3:
                for k, v in delta_ckpt.items():
                    if 'llama_model.model.embed_tokens.weight' in k:
                        checkpoint['llama_model.base_model.model.model.embed_tokens.weight'] = v
                    elif 'llama_model.lm_head.weight' in k:
                        checkpoint['llama_model.base_model.model.lm_head.weight'] = v
                    else:
                        checkpoint[k] = v
            else:
                checkpoint = delta_ckpt
            # for param in self.model.parameters():
            #     param.requires_grad = False
            # self.model.load_state_dict(checkpoint, strict=False)

