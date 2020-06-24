from models.digitsum_image import digitsum_image
from models.digitsum_text import digitsum_text

def get_model(args):
    model_instance = _get_model_instance(args.arch)

    print('Fetching model %s - %s' % (args.arch, args.model_name))

    if args.arch == 'digitsum_image':
        model = model_instance(args.model_name)
    elif args.arch == 'digitsum_text':
        model = model_instance(args.model_name)
    else:
        raise 'Model {} not available.'.format(args.arch)

    return model

def _get_model_instance(arch):
    return{
        'digitsum_image': digitsum_image,
        'digitsum_text': digitsum_text,
    }[arch]