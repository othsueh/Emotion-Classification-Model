import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient Loss.
    CCC = [2 * pxy * ax * ay] / [ax² + ay² + (μx - μy)²]
    where:
    - pxy is the Pearson correlation coefficient
    - ax, ay are the standard deviations
    - μx, μy are the means
    
    For loss: loss = 1 - CCC (so we minimize loss to maximize CCC)
    """
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, pred, target):
        # Clone tensors to avoid modifying the originals
        pred = pred.clone()
        target = target.clone()
        
        # Calculate CCC for arousal (dim 0) and valence (dim 1)
        arousal_ccc = self._calculate_ccc(pred[:, 0], target[:, 0])
        valence_ccc = self._calculate_ccc(pred[:, 1], target[:, 1])
        
        # Total loss is negative average CCC (to minimize)
        loss = 1 - (arousal_ccc + valence_ccc) / 2
        
        # Detach CCC values for logging (important to prevent memory leaks)
        arousal_ccc_detached = arousal_ccc.detach()
        valence_ccc_detached = valence_ccc.detach()
        
        return loss, arousal_ccc_detached, valence_ccc_detached
    
    def _calculate_ccc(self, pred, target):
        # Mean and variance calculations
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        
        pred_var = torch.var(pred, unbiased=False)
        target_var = torch.var(target, unbiased=False)
        
        # Covariance calculation
        cov = torch.mean((pred - pred_mean) * (target - target_mean))
        
        # CCC formula
        numerator = 2 * cov
        denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
        
        # Avoid division by zero
        epsilon = 1e-8
        ccc = numerator / (denominator + epsilon)
        
        return ccc


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax Loss that accounts for class imbalance.
    
    Args:
        sample_per_class (list or torch.Tensor): Number of samples per class.
        reduction (str, optional): Specifies the reduction to apply to the output. 
            Options: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    def __init__(self, sample_per_class, reduction='mean'):
        """
        Args:
            sample_per_class: A list or tensor containing the count of samples for each class.
                              This should be the raw count, not percentages.
            reduction: Specifies the reduction to apply to the output.
        """
        super(BalancedSoftmaxLoss, self).__init__()
        self.sample_per_class = torch.tensor(sample_per_class) if not isinstance(sample_per_class, torch.Tensor) else sample_per_class
        self.reduction = reduction
        
    def forward(self, logits, labels):
        """
        Args:
            logits: A float tensor of size [batch, no_of_classes].
            labels: A int tensor of size [batch].
        Returns:
            loss: A float tensor. Balanced Softmax Loss.
        """
        spc = self.sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        adjusted_logits = logits + spc.log()
        loss = F.cross_entropy(input=adjusted_logits, target=labels, reduction=self.reduction)
        return loss

class DiverseExpertLoss(nn.Module):
    def __init__(self, cls_num_list=None,  max_m=0.5, s=30, tau=1.7):
        super().__init__()
        self.base_loss = F.cross_entropy 
     
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau 

    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0

        # Obtain logits from each expert  
        expert1_logits = extra_info['logits'][0]
        expert2_logits = extra_info['logits'][1] 
        expert3_logits = extra_info['logits'][2]  
 
        # Softmax loss for expert 1 
        loss += self.base_loss(expert1_logits, target)
        
        # Balanced Softmax loss for expert 2 
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        loss += self.base_loss(expert2_logits, target)
        
        # Inverse Softmax loss for expert 3
        inverse_prior = self.inverse_prior(self.prior)
        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior+ 1e-9) 
        loss += self.base_loss(expert3_logits, target)
   
        return loss
