caffe.set_mode_gpu();
gpu_id = 3;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
net = caffe.Net(protoFile, modelFile, 'train');