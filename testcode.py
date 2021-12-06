import numpy as np
import torch
overlaps=np.random.random((3,10))
overlaps=torch.tensor(overlaps)
print ('overlaps:',overlaps)
best_prior_overlap, best_prior_idx = overlaps.max(1,keepdim=True)
print("best_prior_overlap:",best_prior_overlap)
print("best_prior_idx:",best_prior_idx)
# ignore hard gt
#排除与groundtruth<0.2的先验框
valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
print("valid_gt_idx:",valid_gt_idx)
best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
print("best_prior_idx_filter:",best_prior_idx_filter)
#if best_prior_idx_filter.shape[0] <= 0:
#    loc_t[idx] = 0
#    conf_t[idx] = 0
#    return
# [1,num_priors] best ground truth for each prior
best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
print("best_truth_overlap:",best_truth_overlap)
print("best_truth_idx:",best_truth_idx)
best_truth_idx.squeeze_(0)
best_truth_overlap.squeeze_(0)
best_prior_idx.squeeze_(1)
best_prior_idx_filter.squeeze_(1)
best_prior_overlap.squeeze_(1)
best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)
print("best_truth_overlap:",best_truth_overlap)
for j in range(best_prior_idx.size(0)):     # 判别此anchor是预测哪一个boxes
    best_truth_idx[best_prior_idx[j]] = j
print("best_truth_idx:",best_truth_idx)

truths=torch.Tensor(np.random.random((3,4)))
matches = truths[best_truth_idx] 
print(matches)

