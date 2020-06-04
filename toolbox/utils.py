import os

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    #plt.show(block = False)
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")

# check if folder exists, otherwise create it
def check_dir(dir_path):
    dir_path = dir_path.replace('//','/')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def count_params(model):
   return sum([p.data.nelement() for p in model.parameters()])

def compute_batch(batch, args, model):
    target_list = []
    output_list = []
    batch_size=len(batch)
    target_size=batch[0][1].size()[0]
    for (input, target) in batch:
        input, target = input.to(args.device), target.to(args.device)
        target_list.append(target)
        output = model(input)
        output_list.append(output)
    output_batch = torch.stack(output_list, dim=0)
    target_batch = torch.stack(target_list, dim=0)
    output_batch = output_batch.squeeze(1)
    target_batch = target_batch.squeeze(1)
    return output_batch, target_batch
