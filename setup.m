% set your mat caffe path
matcaffePath = '/media/SecondDisk/yanpengxiang/Instance-Saliency/MSRNet/deeplab-caffe/matlab/';
addpath(matcaffePath)
addpath(genpath('.'))

% default: using GoogleNet
% other option: VGG16 which is used in the paper
%param = getParam('GoogleNet');  
param = getParam('MSRN');

net = initModel(param);
