import torch
import mrpl
from IPython import embed
import argparse


def MRPL_2_images(im,ref,mode='mrpl',verbose=True):

    '''
    Simple code to compute the MR_perceptual loss between two images:
    
    Input:
        - im : image path of the image to compare
        - ref : image path of the image to compare
        - mode (optional) : loss model, mrpl (best results) or mr_simple (best setup with all layers)
        - verbose (optional) : print more informations

    Output:
        - ex_d0 : loss between im and ref
    '''

    use_gpu = False         # Whether to use GPU
    spatial = False         # Return a spatial map of perceptual distance.

    if mode == 'mrpl':
        loss_fn = mrpl.MRPL(net='alex', spatial=spatial,mrpl=True,verbose=verbose) 
    elif mode == 'mr_simple':
        loss_fn = mrpl.MRPL(net='alex', spatial=spatial,mrpl=False,loss_type='CE',norm='sigmoid',feature='linear',resolution=['x1','x2'],mrpl_like=True,verbose=verbose) 
    else :
        raise('Not implemented !')

    if(use_gpu):
        loss_fn.cuda()

    ## Example usage with dummy tensors
    '''dummy_im0 = torch.zeros(1,3,64,64) # image should be RGB, normalized to [-1,1]
    dummy_im1 = torch.zeros(1,3,64,64)
    if(use_gpu):
        dummy_im0 = dummy_im0.cuda()
        dummy_im1 = dummy_im1.cuda()
    dist = loss_fn.forward(dummy_im0,dummy_im1)'''

    ## Example usage with images
    ex_ref = mrpl.im2tensor(mrpl.load_image(ref))
    ex_p0 = mrpl.im2tensor(mrpl.load_image(im))


    if(use_gpu):
        ex_ref = ex_ref.cuda()
        ex_p0 = ex_p0.cuda()

    ex_d0 = loss_fn.forward(ex_ref,ex_p0)

    return ex_d0

    '''else:
        print('Distances: (%.3f, %.3f)'%(ex_d0.mean(), ex_d1.mean()))            # The mean distance is approximately the same as the non-spatial distance
        
        # Visualize a spatially-varying distance map between ex_p0 and ex_ref
        import pylab
        pylab.imshow(ex_d0[0,0,...].data.cpu().numpy())
        pylab.show()'''

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mrpl', help='mrpl (best results of the paper) or mr_simple (only x1 and x2 resolution)')
    parser.add_argument('--ref', type=str, default='./imgs/ex_ref.png', help='path to the reference image')
    parser.add_argument('--im1', type=str, default='./imgs/ex_p0.png', help='path to the first image')
    parser.add_argument('--im2', type=str, default='./imgs/ex_p1.png', help='path to the second image')
    opt = parser.parse_args()

    dist1 = MRPL_2_images(opt.im1,opt.ref,mode=opt.mode)
    dist2 = MRPL_2_images(opt.im2,opt.ref,mode=opt.mode)

    print('Distances: (%.3f, %.3f)'%(dist1, dist2))

if __name__=="__main__":
    main()
