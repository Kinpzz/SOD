% read filename
fp = fopen('/media/SecondDisk/yanpengxiang/dataset/MSRN/test_id.txt', 'r');
C = textscan(fp, '%s %*[^\n]');
filenames = C{1};
fclose(fp);

I = imreadRGB(fullfile('/media/SecondDisk/yanpengxiang/dataset/MSRN/image', [filenames{2}, '.jpg']));

imsz = [size(I,1), size(I,2)];

tic;
[P, S, M] = getProposals(I, net, param);
figure();
imshow(M);
res = propOpt(P, S, param);

% scale bboxes to full size
res = bsxfun(@times, res, imsz([2 1 2 1])');

% optional window refining process
resRefine = refineWin(I, res, net, param);
toc
figure();
subplot(1,2,1)
imshow(I)
for i = 1:size(res,2)
    rect = res(:,i);
    rect(3:4) = rect(3:4)-rect(1:2) +1;
    rectangle('Position',rect,'linewidth',2,'edgecolor',[1 0 0]);
end
title('W/O Window Refining');

subplot(1,2,2)

imshow(I)
for i = 1:size(resRefine,2)
    rect = resRefine(:,i);
    rect(3:4) = rect(3:4)-rect(1:2) +1;
    rectangle('Position',rect,'linewidth',2,'edgecolor',[1 0 0]);
end
title(sprintf('With Window Refining\n for Small Objects'))

