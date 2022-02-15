import numpy as np
import mrpl
from data import data_loader as dl
import argparse
from IPython import embed
import optuna

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_mode', type=str, default='2afc', help='[2afc,jnd]')
parser.add_argument('--datasets', type=str, nargs='+', default=['val/traditional','val/cnn','val/superres','val/deblur','val/color','val/frameinterp'], help='datasets to test - for jnd mode: [val/traditional],[val/cnn]; for 2afc mode: [train/traditional],[train/cnn],[train/mix],[val/traditional],[val/cnn],[val/color],[val/deblur],[val/frameinterp],[val/superres]')
#parser.add_argument('--datasets', type=str, nargs='+', default=['val/traditional','val/cnn'], help='datasets to test - for jnd mode: [val/traditional],[val/cnn]; for 2afc mode: [train/traditional],[train/cnn],[train/mix],[val/traditional],[val/cnn],[val/color],[val/deblur],[val/frameinterp],[val/superres]')
parser.add_argument('--model', type=str, default='baseline', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
parser.add_argument('--net', type=str, default='vgg', help='[squeeze], [alex2], or [vgg] for network architectures')
parser.add_argument('--colorspace', type=str, default='Lab', help='[Lab] or [RGB] for colorspace to use for l2, ssim model types')
parser.add_argument('--batch_size', type=int, default=50, help='batch size to test image patches in')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='gpus to use')
parser.add_argument('--nThreads', type=int, default=4, help='number of threads to use in data loader')
parser.add_argument('--gram', action='store_true', help='model was initialized from scratch')
parser.add_argument('--RBF', action='store_true', help='model was initialized from scratch')
parser.add_argument('--randomRBF', action='store_true', help='model was initialized from scratch')
parser.add_argument('--entropy', action='store_true', help='model was initialized from scratch')
parser.add_argument('--model_path', type=str, default=None, help='location of model, will default to ./weights/v[version]/[net_name].pth')

parser.add_argument('--from_scratch', action='store_true', help='model was initialized from scratch')
parser.add_argument('--train_trunk', action='store_true', help='model trunk was trained/tuned')
parser.add_argument('--version', type=str, default='0.1', help='v0.1 is latest, v0.0 was original release')

opt = parser.parse_args()
nbslice=15
if(opt.RBF ):
	opt.gram, opt.model = False, 'baseline'
	opt.randomRBF = False
	opt.batch_size = 1
if(opt.randomRBF ):
	opt.gram, opt.model = False, 'baseline'
	opt.RBF = False

if((opt.model in ['l2','ssim','pattern_spectrum']) or opt.gram ):
	opt.batch_size = 1
if(opt.net=='squeeze' ):
	nbslice = 7

# initialize model
trainer = mrpl.Trainer()
# trainer.initialize(model=opt.model,net=opt.net,colorspace=opt.colorspace,model_path=opt.model_path,use_gpu=opt.use_gpu)
trainer.initialize(model=opt.model, net=opt.net, colorspace=opt.colorspace, 
	model_path=opt.model_path, use_gpu=opt.use_gpu, pnet_rand=opt.from_scratch, pnet_tune=opt.train_trunk,gram=opt.gram,entropy=opt.entropy,RBF=opt.RBF,randomRBF=opt.randomRBF,
	version=opt.version, gpu_ids=opt.gpu_ids)

if(opt.model in ['net-lin','net']):
	print('Testing model [%s]-[%s]'%(opt.model,opt.net))
elif(opt.model in ['l2','ssim', 'pattern_spectrum']):
	print('Testing model [%s]-[%s]'%(opt.model,opt.colorspace))
all_results= {}
for dataset in opt.datasets:
	all_results[dataset]={}
# initialize data loader
for dataset in opt.datasets:
	if opt.model != 'pattern_spectrum_net':data_loader = dl.CreateDataLoader(dataset,dataset_mode=opt.dataset_mode, batch_size=opt.batch_size, nThreads=opt.nThreads)
	else:data_loader = dl.CreateDataLoader(dataset,dataset_mode=opt.dataset_mode,morpho=True, batch_size=opt.batch_size, nThreads=opt.nThreads)
	# evaluate model on data
	if (opt.dataset_mode == '2afc'):

		if opt.model != 'pattern_spectrum_net':(score,scorelayers, results_verbose) = mrpl.score_2afc_datasetvisu(data_loader, trainer, name=dataset,nbslice=nbslice)
		else:(score,scorelayers, results_verbose) = mrpl.score_2afc_datasetvisu(data_loader, trainer,morpho=True, name=dataset,nbslice=nbslice)
	elif(opt.dataset_mode=='jnd'):
		(score,scorelayers, results_verbose) = mrpl.score_jnd_dataset(data_loader, trainer.forward, name=dataset)

	# print results
	print('  Dataset [%s]:  score paper %.2f '%(dataset,100.*score))
	all_results[dataset]=results_verbose
	for i in range(nbslice):
		namekey = 'layer_' + str(i)
		print('score of '+namekey,100.*scorelayers[namekey])


print(all_results)
def objective(trial):
	x0 = trial.suggest_int('x0', 0, 1)
	x1 = trial.suggest_int('x1', 0, 1)
	x2 = trial.suggest_int('x2', 0, 1)
	x3 = trial.suggest_int('x3', 0, 1)
	x4 = trial.suggest_int('x4', 0, 1)
	x5 = trial.suggest_int('x5', 0, 1)
	x6 = trial.suggest_int('x6', 0, 1)
	x7 = trial.suggest_int('x7', 0, 1)
	x8 = trial.suggest_int('x8', 0, 1)
	x9 = trial.suggest_int('x9', 0, 1)
	x10 = trial.suggest_int('x10', 0, 1)
	x11 = trial.suggest_int('x11', 0, 1)
	x12 = trial.suggest_int('x12', 0, 1)
	x13 = trial.suggest_int('x13', 0, 1)
	x14 = trial.suggest_int('x14', 0, 1)

	#x5 = trial.suggest_int('x5', 1, 3)
	#x6 = trial.suggest_int('x6', 1, 3)
	scores_mean=0
	for dataset in opt.datasets:
		results1dataset = all_results[dataset]
		layers0 = results1dataset['layers0']
		layers1 = results1dataset['layers1']
		gts = results1dataset['gts']
		n0 = 'layer_0'
		n1 = 'layer_1'
		n2 = 'layer_2'
		n3 = 'layer_3'
		n4 = 'layer_4'
		n5 = 'layer_5'
		n6 = 'layer_6'
		n7 = 'layer_7'
		n8 = 'layer_8'
		n9 = 'layer_9'
		n10 = 'layer_10'
		n11 = 'layer_11'
		n12 = 'layer_12'
		n13 = 'layer_13'
		n14 = 'layer_14'
		'''val0 = (x5*np.array(layers0[n0])+x6*np.array(layers0[n5]))**x0+(x7*np.array(layers0[n1])+x8*np.array(layers0[n6]))**x1 \
			   +(x9*np.array(layers0[n2])+x10*np.array(layers0[n7])) ** x2 +(x11*np.array(layers0[n3])+x12*np.array(layers0[n8]))**x3 \
			   +(x13*np.array(layers0[n4])+x14*np.array(layers0[n9]))**x4
		val1 = (x5*np.array(layers1[n0])+x6*np.array(layers1[n5]))**x0+(x7*np.array(layers1[n1])+x8*np.array(layers1[n6]))**x1 \
			   +(x9*np.array(layers1[n2])+x10*np.array(layers1[n7])) ** x2 +(x11*np.array(layers1[n3])+x12*np.array(layers1[n8]))**x3 \
			   +(x13*np.array(layers1[n4])+x14*np.array(layers1[n9]))**x4'''
		val0 = (x5 * np.array(layers0[n0]) + x6 * np.array(layers0[n5]) + x0 * np.array(layers0[n10]))  + (x7 * np.array(layers0[n1]) + x8 * np.array(layers0[n6]) +x1 * np.array(layers0[n11]) )  \
			   + (x9 * np.array(layers0[n2]) + x10 * np.array(layers0[n7]) + x2*  np.array(layers0[n12]) )  + (x11 * np.array(layers0[n3]) + x12 * np.array(layers0[n8]) + x3*  np.array(layers0[n13]) )  \
			   + (x13 * np.array(layers0[n4]) + x14 * np.array(layers0[n9]) + x4*  np.array(layers0[n14]) )
		val1 = (x5 * np.array(layers1[n0]) + x6 * np.array(layers1[n5]) + x0 * np.array(layers1[n10]))  + (x7 * np.array(layers1[n1]) + x8 * np.array(layers1[n6]) +x1 * np.array(layers1[n11]) )  \
			   + (x9 * np.array(layers1[n2]) + x10 * np.array(layers1[n7]) + x2*  np.array(layers1[n12]) )  + (x11 * np.array(layers1[n3]) + x12 * np.array(layers1[n8]) + x3*  np.array(layers1[n13]) )  \
			   + (x13 * np.array(layers1[n4]) + x14 * np.array(layers1[n9]) + x4*  np.array(layers1[n14]) )
		d0s = val0# np.array(val0)
		d1s = val1#np.array(val1)
		gts = np.array(gts)
		scores = (d0s < d1s) * (1. - gts) + (d1s < d0s) * gts + (d1s == d0s) * .5
		scores_mean+=np.mean(scores)/len(opt.datasets)
	return -scores_mean


study = optuna.create_study()
study.optimize(objective, n_trials=100)

print(study.best_params)  # E.g. {'x': 2.002108042}
