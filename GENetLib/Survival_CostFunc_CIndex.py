import torch


def R_set(x):
	'''Create an indicator matrix of risk sets, where T_j >= T_i.
	Note that the input data have been sorted in descending order.
	Input:
		x: a PyTorch tensor that the number of rows is equal to the number of samples.
	Output:
		indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
	'''
	n_sample = x.size(0)
	matrix_ones = torch.ones(n_sample, n_sample)
	indicator_matrix = torch.tril(matrix_ones) #下三角阵
	return(indicator_matrix)

def neg_par_log_likelihood(pred, ytime, yevent):
	'''Calculate the average Cox negative partial log-likelihood.
	Note that this function requires the input data have been sorted in descending order.
	Input:
		pred: linear predictors from trained model.
		ytime: true survival time from load_data().
		yevent: true censoring status from load_data().
	Output:
		cost: the cost that is to be minimized.
	'''
	n_observed = yevent.sum(0)
	ytime_indicator = R_set(ytime)
	###if gpu is being used
	if torch.cuda.is_available():
		ytime_indicator = ytime_indicator.cuda()
	###
	risk_set_sum = ytime_indicator.mm(torch.exp(pred))
	diff = pred - torch.log(risk_set_sum)
	sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
	cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
	return(cost)

def c_index(pred, ytime, yevent):
	'''Calculate concordance index to evaluate models.
	Input:
		pred: linear predictors from trained model.
		ytime: true survival time from load_data().
		yevent: true censoring status from load_data().
	Output:
		concordance_index: c-index (between 0 and 1).
	'''
	n_sample = len(ytime)
	ytime_indicator = R_set(ytime)
	ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
	###T_i is uncensored
	censor_idx = (yevent == 0).nonzero()
	zeros = torch.zeros(n_sample)
	ytime_matrix[censor_idx, :] = zeros
	###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
	pred_matrix = torch.zeros_like(ytime_matrix)
	for j in range(n_sample):                                                                 
		for i in range(n_sample):
			if pred[i] < pred[j]:
				pred_matrix[j, i]  = 1
			elif pred[i] == pred[j]: 
				pred_matrix[j, i] = 0.5
	
	concord_matrix = pred_matrix.mul(ytime_matrix) 
	###numerator 
	concord = torch.sum(concord_matrix)  #预测正确的对数
	###denominator
	epsilon = torch.sum(ytime_matrix)  #总共可比较的对数，torch.sum为矩阵的所有元素相加
	###c-index = numerator/denominator
	concordance_index = torch.div(concord, epsilon)
	###if gpu is being used
	if torch.cuda.is_available():
		concordance_index = concordance_index.cuda()
	return(concordance_index)

