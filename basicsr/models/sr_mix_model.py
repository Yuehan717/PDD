from asyncio.log import logger
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.archs.swinir_arch import SwinIR


@MODEL_REGISTRY.register()
class SRMixModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRMixModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        # self.net_g = torch.compile(build_network(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # print("Using gradient clipping",self.opt['train'].get('clip', None))

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.is_use_sharpened_gt_in_pixel= train_opt.get('is_use_sharpened_gt_in_pixel', False)
        self.is_use_sharpened_gt_in_percep= train_opt.get('is_use_sharpened_gt_in_percep', False)

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.cri_pix_name = train_opt['pixel_opt']['type']
        else:
            self.cri_pix = None
        
        if train_opt.get('grad_opt'):
            self.cri_grad = build_loss(train_opt['grad_opt']).to(self.device)
            self.cri_grad_name = train_opt['grad_opt']['type']
        else:
            self.cri_grad = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        
        if train_opt.get('domain_opt'):
            self.cri_domain = build_loss(train_opt['domain_opt']).to(self.device)
        else:
            self.cri_domain = None
        
        # if self.cri_pix is None and self.cri_perceptual is None:
        #     raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, s_data, t_data):
        if 'lq' in s_data:
            self.lq_s = s_data['lq'].to(self.device)
        if 'lq' in t_data:
            self.lq_t = t_data['lq'].to(self.device)
        if 'gt' in s_data:
            self.gt_s = s_data['gt'].to(self.device)
        if 'gt' in t_data:
            self.gt_t = t_data['gt'].to(self.device)# Use gt or gt_unsharp
    
    def feed_test_data(self, data):
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)  # Use gt or gt_unsharp
       
    def optimize_parameters(self, current_iter):
        # logger = get_root_logger()
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # logger.info(f'Optimization of iter {current_iter} starts!')
        self.optimizer_g.zero_grad()
        self.output_s, self.feats_s = self.net_g(self.lq_s)
        self.output_t, self.feats_t = self.net_g(self.lq_t)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_domain:
            self.cri_domain.train()
            l_domain = self.cri_domain(self.feats_s, self.feats_t)
            if l_domain is not None:
                l_total += l_domain
                loss_dict['l_domain'] = l_domain
                
        if self.cri_pix:
            l_pix = self.cri_pix(self.output_s, self.gt_s)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # Grad loss
        if self.cri_grad:
            l_grad = self.cri_grad(self.output, self.gt_s)
            l_total += l_grad
            loss_dict['l_grad'] = l_grad
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt_s)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        
        # start.record()
        l_total.backward()
        # logger.info(f'Backward of iter {current_iter} finishes!')
        # end.record()
        # torch.cuda.synchronize()
        # logger.warning(f"backward time is {start.elapsed_time(end)}")
        if self.opt['train'].get('clip', None) is not None:
            # logger.info('Using gradient clip')
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.opt['train']["clip"])
        # self.print_grad()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # print('before')
        if self.opt['network_g']['type'].lower()=='swinir':
            # print('inside')
            _, _, h_old, w_old = self.lq.size()
            h_pad = (h_old // 8 + 1) * 8 - h_old
            w_pad = (w_old // 8 + 1) * 8 - w_old
            self.lq = torch.cat([self.lq, torch.flip(self.lq, [2])], 2)[:, :, :h_old + h_pad, :]
            self.lq = torch.cat([self.lq, torch.flip(self.lq, [3])], 3)[:, :, :, :w_old + w_pad]
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
        if self.opt['network_g']['type'].lower()=='swinir':
            self.output = self.output[..., :h_old * 4, :w_old * 4]
            
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_test_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def print_grad(self):
        logger = get_root_logger()
        for name,param in self.net_g.named_parameters():
            logger.info(f'Layer {name}: {param.grad}')
