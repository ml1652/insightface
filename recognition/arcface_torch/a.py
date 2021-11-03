criterion = torch.nn.CrossEntropyLoss()
for epoch in range(start_epoch, cfg.num_epoch):
    train_sampler.set_epoch(epoch)
    for step, (img, label, idx) in enumerate(train_loader):  # image.shape=[]
        global_step += 1
        features = F.normalize(backbone(img))
        #load deepface genereated labels
        json_dir = "/scratch/ml1652/code/insightface/recognition/arcface_torch/ms1m-retinaface-t1_deepfaceLable/%d.json" % idx
        with open(json_dir, 'r') as f:
            non_id_label = json.load(f)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        output_netD = netD(features)
        errD_real = criterion(output_netD, non_id_label)
        errD_real.backward()
        opt_netD.step()
        D_x = output_netD.mean().item()
        output_netD = netD(features)
        errD_real = criterion(output_netD, non_id_label)
        errD_neg = -errD_real
        errD_neg.backward()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)
        if cfg.fp16:  # 因为是半精度，当模型接近于收敛的时候，模型梯度变小，出现下溢出
            features.backward(grad_amp.scale(x_grad))  # 因为半精度的数值范围有限，因此需要将loss 乘以一个scale.用它放大
            grad_amp.unscale_(opt_backbone)
            # 梯度剪裁，求所有参数的二范数，如果大于max_norm,都乘以max_norm所有参数的二范数
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
            grad_amp.step(opt_backbone)
            grad_amp.update()
        else:
            features.backward(x_grad)  # 分段进行求导，因为partialfc将类中心存在不同显卡中，需要手动求导
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