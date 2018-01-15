% refer to
% https://github.com/ShaoqingRen/faster_rcnn/blob/master/functions/rpn/proposal_train.m
% for more api details
clear;clc;close all;
% init
matcaffePath = '/media/SecondDisk/yanpengxiang/Instance-Saliency/MSRNet/deeplab-caffe/matlab/';
addpath(matcaffePath)
addpath(genpath('./'));

opts = config();

%caffe init
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(opts.gpu_id);
caffe_solver = caffe.Solver(fullfile(opts.model_path, 'solver.prototxt'));
caffe_solver.net.copy_from(fullfile(opts.model_path, 'MSRN_iter_3499.caffemodel'));

% load dataset
load imgIdxWithBBoxBinaryLabels.mat;
load(fullfile(opts.dataset_root, 'train_val_id.mat'));
load center100.mat;
train_set = imgIdx(train_id);
val_set = imgIdx(val_id);

% train
val_losses = do_train(train_set, val_set, caffe_solver, opts);
save val_losses;
caffe.reset_all();