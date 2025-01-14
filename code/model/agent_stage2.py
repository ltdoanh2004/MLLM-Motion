import os.path

import torch
from header import *
import os
import torch.optim as optim
# from model.MotionDiffuse.datasets1 import Text2MotionDataset
# from model.MotionDiffuse.datasets1 import build_dataloader
# import model.MotionDiffuse.utils.paramUtil as paramUtil
# from model.MotionDiffuse.options.train_options import TrainCompOptions
from model.MotionDiffuse.utils.plot_script import *
import os
from os.path import join as pjoin
from model.MotionDiffuse.models import MotionTransformer
# from model.MotionDiffuse.trainers import DDPMTrainer
from model.MotionDiffuse.utils.utils import print_current_loss
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
            lr=0.0004,
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

        
            
        @torch.no_grad()
        def predict(self):
            self.ds_engine.module.eval()
            output = self.ds_engine.generate(self.args)
            return output

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()

        loss, mle_acc, mse_loss = self.ds_engine(batch)
        # loss, mle_acc, mse_loss = self.model(batch)
        self.writer.add_scalar('loss', loss, current_step)
        self.writer.add_scalar('mle_acc', mle_acc, current_step)
        # if isinstance(mse_loss, list):
        #     self.writer.add_scalar('img_mse_loss', mse_loss[0], current_step)
        #     self.writer.add_scalar('vid_mse_loss', mse_loss[1], current_step)
        #     self.writer.add_scalar('aud_mse_loss', mse_loss[2], current_step)
        if isinstance(mse_loss, torch.Tensor):
            self.writer.add_scalar('mse_loss', mse_loss, current_step)
        else:
            pass
        # self.writer.add_scalar('mse_loss', mse_loss, current_step)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        # pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}; mse_loss: {round(mse_loss[0].item(), 4)} ')
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(
                f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
            # ; mse_loss: {round(mse_loss[0].item(), 4)}
            with open('gpt_output_new.txt', 'a') as f:
                f.write(f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}' + '\n')
        mle_acc *= 100
        return mle_acc
    
    

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

