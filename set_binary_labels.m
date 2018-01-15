clear;
% load image
dataset_root = 'G:\dataset\SOS\';
load(fullfile(dataset_root, 'imgIdxWithBBoxTrainingOnly.mat'));
load center100.mat;
imgIdx=imgIdx(1294:end);

% first select the nearest bbox to gt in 4D dimensions
for i = 1:length(imgIdx)
    img_info = imfinfo(fullfile(dataset_root, 'img', imgIdx(i).name));
    bbox_anno=imgIdx(i).anno';
    imsz = [img_info.Height, img_info.Width];
    imgIdx(i).bbox_id = find_nearst_bbox(bbox_anno, imsz, center); 
    if mod(i,20) == 0
        fprintf('%d times\n', i);
    end
end

% input bbox annotation
% output nearst center exampler bbox
function [ nearst_bbox_id ] = find_nearst_bbox( bbox_anno, imsz , center)
bbox_num = size(bbox_anno,2);
nearst_bbox_id = zeros(1, bbox_num);
norm_bbox = bbox_anno ./ repmat(imsz([2 1 2 1])', 1, bbox_num);
for i = 1:bbox_num
    result = norm_bbox(:,i);
    distance = sum((center - repmat(result,1,100)).^2, 1);
    [~, nearst_bbox_id(i)] = min(distance(:));
end
end

