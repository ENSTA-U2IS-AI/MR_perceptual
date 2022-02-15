import numpy as np
import mrpl
from data import data_loader as dl
import argparse
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_mode', type=str, default='2afc', help='[2afc,jnd]')
parser.add_argument('--datasets', type=str, nargs='+', default=['val/traditional','val/cnn','val/superres','val/deblur','val/color','val/frameinterp'], help='datasets to test - for jnd mode: [val/traditional],[val/cnn]; for 2afc mode: [train/traditional],[train/cnn],[train/mix],[val/traditional],[val/cnn],[val/color],[val/deblur],[val/frameinterp],[val/superres]')
#parser.add_argument('--datasets', type=str, nargs='+', default=['val/traditional','val/cnn'], help='datasets to test - for jnd mode: [val/traditional],[val/cnn]; for 2afc mode: [train/traditional],[train/cnn],[train/mix],[val/traditional],[val/cnn],[val/color],[val/deblur],[val/frameinterp],[val/superres]')
parser.add_argument('--model', type=str, default='baseline', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
parser.add_argument('--net', type=str, default='vgg', help='[squeeze], [alex], or [vgg] for network architectures')
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
nbslice=5
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
	for i in range(nbslice):
		namekey = 'layer_' + str(i)
		print('score of '+namekey,100.*scorelayers[namekey])

