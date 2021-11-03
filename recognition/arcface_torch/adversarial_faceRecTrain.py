import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

import losses
from backbones import get_model
from dataset import MXFaceDataset, SyntheticDataset, DataLoaderX
from partial_fc import PartialFC
from utils.utils_amp import MaxClipGradScaler
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
import json
from discriminator import CelebaRegressor

# -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=2234



def main(args):
    cfg = get_config(args.config)
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist.init_process_group('gloo')
    except KeyError:
        world_size = 1
        rank = 0
        dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:12584", rank=rank, world_size=world_size)

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    if cfg.rec == "synthetic":
        train_set = SyntheticDataset(local_rank=local_rank)
    else:
        train_set = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=train_set, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(local_rank)
    num_input_features = 512
    netD = CelebaRegressor(num_input_features).cuda()

    if cfg.resume:
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank == 0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            if rank == 0:
                logging.info("resume fail, backbone init successfully!")

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    netD = torch.nn.parallel.DistributedDataParallel(
        module=netD, broadcast_buffers=False, device_ids=[local_rank])
    netD.train()

    margin_softmax = losses.get_loss(cfg.loss)
    module_partial_fc = PartialFC(
        rank=rank, local_rank=local_rank, world_size=world_size, resume=cfg.resume,
        batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_pfc = torch.optim.SGD(
        params=[{'params': module_partial_fc.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_netD = torch.optim.SGD(
        params=[{'params': netD.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    num_image = len(train_set)
    total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = num_image // total_batch_size * cfg.warmup_epoch
    cfg.total_step = num_image // total_batch_size * cfg.num_epoch

    def lr_step_func(current_step):
        cfg.decay_step = [x * num_image // total_batch_size for x in cfg.decay_epoch]
        if current_step < cfg.warmup_step:
            return current_step / cfg.warmup_step
        else:
            return 0.1 ** len([m for m in cfg.decay_step if m <= current_step])

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR( #lr_scheduler用来调增学习率
        optimizer=opt_backbone, lr_lambda=lr_step_func)
    scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_pfc, lr_lambda=lr_step_func)
    scheduler_netD = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_netD,lr_lambda=lr_step_func)

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    val_target = cfg.val_targets
    callback_verification = CallBackVerification(2000, rank, val_target, cfg.rec)
    callback_logging = CallBackLogging(50, rank, cfg.total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    start_epoch = 0
    global_step = 0
    #混合精度：在训练最开始之前实例化一个GradS对象初始化grad_amp，混合精度训练的初始化
    grad_amp = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None

    criterion = torch.nn.CrossEntropyLoss()
    attribute_D = "happy"
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for step, (img, label, idx) in enumerate(train_loader): #image.shape=[]
            global_step += 1
            features = F.normalize(backbone(img))
            non_id_label_batch = []
            for i in idx:
                json_dir = "/scratch/ml1652/code/insightface/recognition/arcface_torch/ms1m-retinaface-t1_deepfaceLable/%d.json"%i
                with open(json_dir, 'r') as f:
                    non_id_label = json.load(f)
                non_id_label_batch.append(non_id_label["emotion"][attribute_D])

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            #Pass features into discriminator
            #Compute cross entropy loss using our labels
            #opt_discriminator.zero_grad()
            #Call disc_loss.backward()
            #Call opt_discriminator.step()
            netD.zero_grad()
            output_netD =netD(features)
            errD_real = criterion(output_netD, non_id_label_batch)
            errD_real.backward()
            opt_netD.step()
            D_x = output_netD.mean().item()


            # We don't want the discriminator to improve by changing the backbone
            #opt_backbone.zero_grad()
            # Pass features into discriminator
            # Compute cross entropy loss using our labels
            # disc_loss = -disc_loss
            # Call disc_loss.backward()
            # The backbone gradients will now contain information
            # on how to make the discriminator's performance worse
            output_netD =netD(features)
            errD_real = criterion(output_netD, non_id_label_batch)
            errD_neg = -errD_real
            errD_neg.backward()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)
            if cfg.fp16: #因为是半精度，当模型接近于收敛的时候，模型梯度变小，出现下溢出
                features.backward(grad_amp.scale(x_grad)) #因为半精度的数值范围有限，因此需要将loss 乘以一个scale.用它放大
                grad_amp.unscale_(opt_backbone)
                #梯度剪裁，求所有参数的二范数，如果大于max_norm,都乘以max_norm所有参数的二范数
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                grad_amp.step(opt_backbone)
                grad_amp.update()
            else:
                features.backward(x_grad) #分段进行求导，因为partialfc将类中心存在不同显卡中，需要手动求导
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                # This step now improves arcface loss and also reduces discriminator performance
                opt_backbone.step()

            opt_pfc.step()
            module_partial_fc.update()
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, cfg.fp16, scheduler_backbone.get_last_lr()[0], grad_amp)
            callback_verification(global_step, backbone)
            scheduler_backbone.step()
            scheduler_pfc.step()
        callback_checkpoint(global_step, backbone, module_partial_fc)
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())
