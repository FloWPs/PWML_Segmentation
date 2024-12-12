from easydict import EasyDict

cfg = EasyDict()
cfg.batch_size = 4 #16 #MODIF FLORA
cfg.epoch = 200
cfg.learning_rate = 1e-4 #1e-4 # modif flora
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.patience = 15 # modif flora
cfg.inference_threshold = 0.7 # modif flora

cfg.transunet = EasyDict()
cfg.transunet.img_dim = 128 #256 #128 # modif flora
cfg.transunet.in_channels = 3
cfg.transunet.out_channels = 128 
cfg.transunet.head_num = 4
cfg.transunet.mlp_dim = 512
cfg.transunet.block_num = 8
cfg.transunet.patch_dim = 16
cfg.transunet.class_num = 1
