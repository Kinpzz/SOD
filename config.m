function opts = config()
opts.dataset_root = '/media/SecondDisk/yanpengxiang/dataset/SOS/';
opts.model_path = '/media/SecondDisk/yanpengxiang/Instance-Saliency/pipeline/SOD/model/Saliency/';
opts.output_dir = '/media/SecondDisk/yanpengxiang/Instance-Saliency/pipeline/SOD/model/save/VGG16/';
opts.batch_size = 5;
opts.width = 256;
opts.height = 256;
opts.imageMean = single(repmat(reshape([103.939 116.779 123.68],1,1,3),...
    [opts.height, opts.width, 1]));
opts.max_iter = 50000;
opts.val_interval = 1000;
opts.do_val = 1;
opts.display = 20;
opts.gpu_id = 2;
opts.snapshot=1000;
end
