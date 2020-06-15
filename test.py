from torch.utils.tensorboard import SummaryWriter
from data_loader import *
from model import *
import os

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(comment='ResNet34')
checkpoint_dir = './results'
ckpt_model_filename = 'ckpt_valid_acc_88.56999969482422_epoch_180.pth'
PATH = os.path.join(checkpoint_dir, ckpt_model_filename)

NUM_CLASSES = 10

# Other
GRAYSCALE = True


def compute_acc(model, data_loader, device):
	correct_pred, num_examples = 0, 0
	model.eval()
	for i, (features, targets) in enumerate(data_loader):
		features = features.to(device)
		targets = targets.to(device)

		logits, probas = model(features)
		_, predicted_labels = torch.max(probas, 1)
		num_examples = targets.size(0)
		assert predicted_labels.size() == targets.size()
		correct_pred = (predicted_labels == targets).sum()
		# print('num_examples', num_examples)
		break
	return correct_pred.float() / num_examples * 100

if __name__ == '__main__':
	model = resnet34(NUM_CLASSES, GRAYSCALE)
	model.load_state_dict(torch.load(PATH))
	model.to(device)
	model = model.eval()
	for batch_idx, (features, targets) in enumerate(test_loader):

		with torch.set_grad_enabled(False):

			test_acc = compute_acc(model, test_loader, device)
			print("\nTest accuracy: %.2f%%" %(test_acc))
			# writer.add_scalar('Test accuracy', test_acc)
			ckpt_model_filename = 'ckpt_test_acc_{}.pth'.format(test_acc)
			ckpt_model_path = os.path.join(checkpoint_dir, ckpt_model_filename)  # model_save
			torch.save(model.state_dict(), ckpt_model_path)
			print("\nDone, save model at {}", ckpt_model_path)

		break
