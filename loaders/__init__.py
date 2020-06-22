from loaders.digitsum_image_loader import DigitSumImageLoader
from loaders.digitsum_text_loader import DigitSumTextLoader

def get_loader(args):
    """get_loader
    :param name:
    """
    return {
        'digitsum_image' : DigitSumImageLoader,
        'digitsum_text' : DigitSumTextLoader,
    }[args.dataset]