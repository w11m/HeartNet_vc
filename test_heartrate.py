import argparse
import torch
from utilities.utils import model_parameters, compute_flops
from utilities.train_eval_classification import validate
import os
import numpy as np
from data_loader.classification.imagenet import val_loader as loader
from data_loader.classification import heart
from utilities.print_utils import *

# ============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"


# ============================================


def main(args):
    # create model
    if args.model == 'dicenet':
        from model.classification import dicenet as net
        model = net.CNNModel(args)
    elif args.model == 'espnetv2':
        from model.classification import espnetv2 as net
        model = net.EESPNet(args)
    elif args.model == 'shufflenetv2':
        from model.classification import shufflenetv2 as net
        model = net.CNNModel(args)
    else:
        NotImplementedError('Model {} not yet implemented'.format(args.model))
        exit()

    num_params = model_parameters(model)
    flops = compute_flops(model)
    print_info_message('FLOPs: {:.2f} million'.format(flops))
    print_info_message('Network Parameters: {:.2f} million'.format(num_params))

    if not args.weights:
        print_info_message('Grabbing location of the ImageNet weights from the weight dictionary')
        from model.weight_locations.classification import model_weight_map

        weight_file_key = '{}_{}'.format(args.model, args.s)
        assert weight_file_key in model_weight_map.keys(), '{} does not exist'.format(weight_file_key)
        args.weights = model_weight_map[weight_file_key]

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus >= 1 else 'cpu'
    weight_dict = torch.load(args.weights, map_location=torch.device(device))
    model.load_state_dict(weight_dict)

    if num_gpus >= 1:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    # Data loading code
    def load_npy(npy_path):
        try:
            npdata = np.load(npy_path).item()
        except:
            npdata = np.load(npy_path)
        return npdata

    def loadData(data_path):
        npy_data = load_npy(data_path)
        signals = npy_data['signals']
        gts = npy_data['gts']
        return signals, gts

    ht_img_width, ht_img_height = args.inpSize, args.inpSize
    ht_batch_size = 5
    signal_length = args.channels
    signals_val, gts_val = loadData('./data_train/fps7_sample10_2D_val_96.npy')
    from data_loader.classification.heart import HeartDataGenerator
    heart_val_data = HeartDataGenerator(signals_val, gts_val, ht_batch_size)
    val_loader = torch.utils.data.DataLoader(heart_val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                             num_workers=args.workers)
    validate(val_loader, model, criteria=None, device=device)


if __name__ == '__main__':
    from commons.general_details import classification_models, classification_datasets

    parser = argparse.ArgumentParser(description='Testing efficient networks')
    parser.add_argument('--workers', default=1, type=int, help='number of data loading workers (default: 4)')
    # parser.add_argument('--data', default='', help='path to dataset')
    parser.add_argument('--dataset', default='imagenet', help='Name of the dataset', choices=classification_datasets)
    parser.add_argument('--batch-size', default=1, type=int, help='mini-batch size (default: 512)')
    parser.add_argument('--num-classes', default=1, type=int, help='# of classes in the dataset')
    parser.add_argument('--s', default=0.2, type=float, help='Width scaling factor')
    parser.add_argument('--weights', type=str,
                        default='/home/tan/William/DiCENeT/EdgeNets/results_classification_main/dgx/model_dicenet_Heart/aug_0.2_1.0/s_0.2_inp_96_sch_hybrid/20190826-152157/dicenet_0.2_best.pth',
                        help='weight file')
    parser.add_argument('--inpSize', default=96, type=int, help='Input size')
    ##Select a model
    parser.add_argument('--model', default='dicenet', choices=classification_models, help='Which model?')
    parser.add_argument('--model-width', default=96, type=int, help='Model width')
    parser.add_argument('--model-height', default=96, type=int, help='Model height')
    parser.add_argument('--channels', default=70, type=int, help='Input channels')

    args = parser.parse_args()
    main(args)
