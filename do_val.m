function loss = do_val(net, dataset, opts)
batch_size = opts.batch_size;
num_samples = length(dataset);
num_batches = ceil(num_samples / batch_size);
val_iter = 0;
loss = 0.0;
for b = 1:num_batches
    Ip = zeros(opts.width,opts.height,3,opts.batch_size);
    labels = zeros(1,1,100,opts.batch_size);
    for i = 1:opts.batch_size
        I = imreadRGB(fullfile(opts.dataset_root, 'img', dataset(val_iter + i).name));
        labels(1,1,dataset(val_iter + i).bbox_id,i) = 1;
        Ip(:,:,:,i) = prepareImage(I, opts);
    end
    % one channel, do not train mask and set to zeros
    mask = zeros(opts.width, opts.height, 1, opts.batch_size);
    net_input = {Ip, mask, labels};
    caffe_net_reshape_as_input(net, net_input);
    net.forward(net_input);
    loss = loss + net.blobs('bbox_binary_loss').get_data();
    val_iter = mod(val_iter + opts.batch_size, num_batches);
end
loss = loss / num_samples;

end