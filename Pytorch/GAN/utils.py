import logging
import numpy as np
import matplotlib.pyplot as plt

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def reshape(x):
    sqrtx = int(np.ceil(np.sqrt(x.shape[1])))
    return x.view(-1, 1, sqrtx, sqrtx)

def showImg(imgs):
    sqrtn = int(np.ceil(np.sqrt(imgs.shape[0])))
    for index, img in enumerate(imgs):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(img.reshape(28, 28), 'gray')
        plt.axis(False)
    plt.show()
# showImg(data_loader.dataset.data[:64])

def set_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler('./log/loss.log')
    logger.addHandler(fh)
    return logger

def plt_loss(G_losses, D_losses):
    plt.plot(np.array(G_losses), 'r', label='G')
    plt.plot(np.array(D_losses), 'b', label='D')
    plt.legend()
    plt.savefig('./log/loss.png')