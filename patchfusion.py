
# from https://gist.github.com/lalunamel/6b582f865d2be881a501c574a136ce69
import torch
from torchvision import transforms
from PIL import Image
import sys
import pathlib
import os
import os.path as osp
import torch
import time
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(pathlib.Path(__file__).parent.resolve(), "../models/PatchFusion"))
sys.path.insert(0, os.path.join(pathlib.Path(__file__).parent.resolve(), "../models/PatchFusion/external"))
#from models.PatchFusion.estimator.models.patchfusion import PatchFusion



def patchFusion(tmp_image_in, tmp_image_out, image_raw_shape = [2160, 3840], patch_split_num = [4, 4]):

    from mmengine.utils import mkdir_or_exist
    from mmengine.config import Config
    from mmengine.logging import MMLogger

    from estimator.utils import RunnerInfo, setup_env, fix_random_seed
    from estimator.models.builder import build_model
    from estimator.datasets.builder import build_dataset
    from estimator.tester import Tester
    from estimator.models.patchfusion import PatchFusion
    from mmengine import print_log
    # args = parse_args()

    image_raw_shape=[int(num) for num in image_raw_shape]
    patch_split_num=[int(num) for num in patch_split_num]
    cai_mode = 'm1'
    process_num = 2
        
    # load config
    config = "models/PatchFusion/configs/patchfusion_depthanything/depthanything_general.py"
    cfg = Config.fromfile(config)
    
    cfg.launcher = 'none'
    cfg_options = {"general_dataloader":{"dataset":{"rgb_image_dir":tmp_image_in}}}
    # or
    # cfg_options = {"general_dataloader.dataset.rgb_image_dir":tmp_image}
    cfg.merge_from_dict(cfg_options)
    work_dir = tmp_image_out
    cfg.work_dir = work_dir
        
    mkdir_or_exist(cfg.work_dir)
    ckp_path = "Zhyever/patchfusion_depth_anything_vitl14"
    cfg.ckp_path = ckp_path
    
    # fix seed
    seed = cfg.get('seed', 5621)
    fix_random_seed(seed)
    
    # start dist training
    if cfg.launcher == 'none':
        distributed = False
        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp.item()))
        rank = 0
        world_size = 1
        env_cfg = cfg.get('env_cfg')
    else:
        distributed = True
        env_cfg = cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl')))
        rank, world_size, timestamp = setup_env(env_cfg, distributed, cfg.launcher)
    
    # build dataloader
    dataloader_config = cfg.general_dataloader
    dataset = build_dataset(cfg.general_dataloader.dataset)
    
    dataset.image_resolution = image_raw_shape
    
    # extract experiment name from cmd
    config_path = config
    exp_cfg_filename = config_path.split('/')[-1].split('.')[0]
    ckp_name = ckp_path.replace('/', '_').replace('.pth', '')
    dataset_name = dataset.dataset_name
    # log_filename = 'eval_{}_{}_{}_{}.log'.format(timestamp, exp_cfg_filename, ckp_name, dataset_name)
    tag = ''
    log_filename = 'eval_{}_{}_{}_{}_{}.log'.format(exp_cfg_filename, tag, ckp_name, dataset_name, timestamp)
    
    # prepare basic text logger
    log_file = osp.join(work_dir, log_filename)
    log_cfg = dict(log_level='INFO', log_file=log_file)
    log_cfg.setdefault('name', timestamp)
    log_cfg.setdefault('logger_name', 'patchstitcher')
    log_cfg.setdefault('file_mode', 'a')
    logger = MMLogger.get_instance(**log_cfg)
    
    # save some information useful during the training
    runner_info = RunnerInfo()
    runner_info.config = cfg # ideally, cfg should not be changed during process. information should be temp saved in runner_info
    runner_info.logger = logger # easier way: use print_log("infos", logger='current')
    runner_info.rank = rank
    runner_info.distributed = distributed
    runner_info.launcher = cfg.launcher
    runner_info.seed = seed
    runner_info.world_size = world_size
    runner_info.work_dir = cfg.work_dir
    runner_info.timestamp = timestamp
    runner_info.save = True
    runner_info.log_filename = log_filename
    runner_info.gray_scale = True
    
    if runner_info.save:
        mkdir_or_exist(work_dir)
        runner_info.work_dir = work_dir
    # log_env(cfg, env_cfg, runner_info, logger)
    
    # build model
    if '.pth' in cfg.ckp_path:
        model = build_model(cfg.model)
        print_log('Checkpoint Path: {}. Loading from a local file'.format(cfg.ckp_path), logger='current')
        if hasattr(model, 'load_dict'):
            print_log(model.load_dict(torch.load(cfg.ckp_path)['model_state_dict']), logger='current')
        else:
            print_log(model.load_state_dict(torch.load(cfg.ckp_path)['model_state_dict'], strict=True), logger='current')
    else:
        print_log('Checkpoint Path: {}. Loading from the huggingface repo'.format(cfg.ckp_path), logger='current')
        assert cfg.ckp_path in \
            ['Zhyever/patchfusion_depth_anything_vits14', 
             'Zhyever/patchfusion_depth_anything_vitb14', 
             'Zhyever/patchfusion_depth_anything_vitl14', 
             'Zhyever/patchfusion_zoedepth'], 'Invalid model name'
        model = PatchFusion.from_pretrained(cfg.ckp_path)
    model.eval()
    
    if runner_info.distributed:
        torch.cuda.set_device(runner_info.rank)
        model.cuda(runner_info.rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[runner_info.rank], output_device=runner_info.rank,
                                                          find_unused_parameters=cfg.get('find_unused_parameters', False))
        logger.info(model)
    else:
        model.cuda()
        
    if runner_info.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        val_sampler = None
    
    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=dataloader_config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=val_sampler)

    # build tester
    tester = Tester(
        config=cfg,
        runner_info=runner_info,
        dataloader=val_dataloader,
        model=model)
    
    tester.run(cai_mode, process_num=process_num, image_raw_shape=image_raw_shape, patch_split_num=patch_split_num)



class PatchFusion:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "patch_split_height": ("INT",{
                    "default": 2,
                    "step":1,
                    "display": "number"
                }),
                "patch_split_width": ("INT",{
                    "default": 2,
                    "step":1,
                    "display": "number"
                }),
                "raw_height": ("INT",{
                    "default": 2160,
                    "step":1,
                    "display": "number"
                }),
                "raw_width": ("INT",{
                    "default": 3840,
                    "step":1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_image",)

    FUNCTION = "run"

    CATEGORY = "PatchFusion"

    def run(self, rgb_image, patch_split_height, patch_split_width, raw_height, raw_width):
        # Define the total pixel counts for SD and SDXL
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_in_path = os.path.join(temp_dir, 'in')
        os.mkdir(temp_in_path)
        temp_out_path = os.path.join(temp_dir, 'out')
        os.mkdir(temp_out_path)

        image_path = os.path.join(temp_in_path, 'rgb_in.png')
        rgb_image = rgb_image.permute(0,3,1,2)
        pil_rgb_image = transforms.ToPILImage()(rgb_image[0])
        pil_rgb_image.save(image_path)
        patchFusion(temp_in_path , temp_out_path, [raw_height, raw_width], [ patch_split_height, patch_split_width])
        # load image
        rgb_out_image = Image.open(os.path.join(temp_out_path, 'rgb_in.png'))
        rgb_out_image.putalpha(1) 
        depth_prediction = transforms.PILToTensor()(rgb_out_image).permute(1,2,0)[None,:,:,:]
        depth_prediction = depth_prediction[:,:,:,:3]/255.0
        return (depth_prediction,)

NODE_CLASS_MAPPINGS = {
    "PatchFusion": PatchFusion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PatchFusion": "PatchFusion depth"
}
