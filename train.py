# -*- coding: utf-8 -*-

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from model import *
from data_loader import *
import os

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(comment='ResNet34')
checkpoint_dir = './results'
dummy_input = torch.ones([128, 1, 28, 28])

##########################
### SETTINGS
##########################

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 200
MIN_TEST_ACC= 79

# Architecture
NUM_FEATURES = 32*32
NUM_CLASSES = 10

# Other
GRAYSCALE = True

def compute_accuracy(model, data_loader, device):
	correct_pred, num_examples = 0, 0
	top2_correct = 0
	top1_acc, top2_acc = 0.0, 0.0
	for i, (features, targets) in enumerate(data_loader):

		features = features.to(device)
		targets = targets.to(device)

		logits, probas = model(features)
		_, predicted_labels = torch.max(probas, 1)
		top2_indices = probas.topk(2, 1)[1]
		targets_trans = targets.t().view(-1,1)
		# print(top2_indices.shape, targets_trans.repeat(1, 2).shape)
		top2_predict = (top2_indices == targets_trans.repeat(1, 2))
		top2_correct += top2_predict.sum()
		num_examples += targets.size(0)
		correct_pred += (predicted_labels == targets).sum()
		top1_acc = correct_pred.float() / num_examples * 100
		top2_acc = top2_correct.float() / num_examples * 100

	return top1_acc, top2_acc



if __name__ == '__main__':

	model = resnet34(NUM_CLASSES, GRAYSCALE)
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
	iter_num = 0
	for epoch in range(NUM_EPOCHS):
		model = model.train()

		for batch_idx, (features, targets) in enumerate(train_loader):
			iter_num += 1
			features = features.to(device)
			targets = targets.to(device)

			### FORWARD AND BACK PROP
			logits, probas = model(features)

			cost = F.cross_entropy(logits, targets)
			writer.add_scalar('Train Loss', cost, iter_num)

			optimizer.zero_grad()

			cost.backward()
			optimizer.step()
			lr = optimizer.param_groups[0]['lr']
			### UPDATE MODEL PARAMETERS
			optimizer.step()

			### LOGGINGpip
			if not batch_idx % 120:
				print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f | lr: %.4f'
					  % (epoch + 1, NUM_EPOCHS, batch_idx,
						 len(train_loader), cost, lr))
		train_acc,_ = compute_accuracy(model, test_loader, device)
		acc_top1, acc_top2 = compute_accuracy(model, test_loader, device)
		writer.add_scalar('Train Acc', train_acc, iter_num)
		writer.add_scalar('Valid Acc', acc_top1, iter_num)
		print('Epoch: %03d/%03d | training accuracy: %.2f%% \nValid accuracy: top1: %.2f%% | top2: %.2f%% ' % (
			epoch + 1, NUM_EPOCHS,
			train_acc, acc_top1, acc_top2))

		model = model.eval()
		with torch.set_grad_enabled(False):

			acc_top1, acc_top2 = compute_accuracy(model, test_loader, device)

			ckpt_model_filename = 'ckpt_valid_acc_{}_epoch_{}.pth'.format(acc_top1, epoch)
			ckpt_model_path = os.path.join(checkpoint_dir, ckpt_model_filename)  # model_save

			if acc_top1 > MIN_TEST_ACC:
				torch.save(model.state_dict(), ckpt_model_path)
				print("\nDone, save model at {}", ckpt_model_path)
				MIN_TEST_ACC = acc_top1
		scheduler.step()

	model = model.to(device)
	with SummaryWriter(comment='ResNet34') as w:
		w.add_graph(model, (dummy_input.to(device),), True)

