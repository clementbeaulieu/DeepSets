from loaders.digitsum_image_loader import DigitSumImageLoader

def get_loader(args):
    """get_loader
    :param name:
    """
    return {
        'digitsum_image' : DigitSumImageLoader,
    }[args.dataset]