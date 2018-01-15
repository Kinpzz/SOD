function val_losses = do_train(train_set, val_set, caffe_solver, opts)

train_set_length = length(train_set);
num_batches = ceil(train_set_length / opts.batch_size);
train_iter = 0;
iter_ = caffe_solver.iter();
loss = 0.0;
val_losses = [];
while (iter_ < opts.max_iter)
    Ip = zeros(opts.width,opts.height,3, opts.batch_size);
    labels = zeros(1,1,100,opts.batch_size);
    for i = 1:opts.batch_size
        I = imreadRGB(fullfile(opts.dataset_root, 'img', train_set(train_iter + i).name));
        labels(1,1,train_set(train_iter + i).bbox_id,i) = 1;
        Ip(:,:,:,i) = prepareImage(I, opts);
    end
    % one channels, do not train mask
    mask = zeros(opts.width, opts.height, 1, opts.batch_size);
    net_input = {Ip, mask, labels};        
    caffe_net_reshape_as_input(caffe_solver.net, net_input);
    caffe_net_set_input_data(caffe_solver.net, net_input);
    caffe_solver.step(1);
    loss = loss + caffe_solver.net.blobs('bbox_binary_loss').get_data();
    % display
    if ~mod(iter_, opts.display)
        fprintf('Training %d times loss: %f\n',iter_, loss/(opts.display*opts.batch_size));
        loss = 0;
    end
    % val 
    if opts.do_val
        if ~mod(iter_, opts.val_interval)
            val_loss = do_val(caffe_solver.net, val_set, opts);
            val_losses = [val_losses val_loss];
            fprintf('Validation loss: %f\n', val_loss);
        end
    end
    % save model
    if ~mod(iter_ + 1, opts.snapshot)
        caffe_solver.net.save(fullfile(opts.output_dir, ['MSRN_iter_', num2str(iter_), '.caffemodel']));
    end
    iter_ = caffe_solver.iter();
    train_iter = mod(train_iter + opts.batch_size, num_batches);
end

end

