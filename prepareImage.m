function Ip = prepareImage(I, param)

Ip = imresize(I(:, :, [3 2 1]),...
        [param.height, param.width], 'bilinear', 'antialiasing', false);
Ip = single(Ip);
Ip = Ip - param.imageMean(1:param.height, 1:param.width, :);
Ip = permute(Ip, [2 1 3]);

end