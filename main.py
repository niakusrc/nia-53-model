import datetime
import sys
import numpy as np
import random
import tensorflow as tf
import models
from TGNet import TGNet
import warnings
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

if __name__ == "__main__":
    print('[*] Program Starts')
    print('Time is : ', datetime.datetime.now())
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0.01)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--drop_p', type=float, default=0.1)
    parser.add_argument('--reg', type=float, default=0.0)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--output_dir', type=str, default='./output/')
    parser.add_argument('--save_dir', type=str, default='./model_saved/')
    parser.add_argument('--model_name', type=str, default='no_named')

    parser.add_argument('--scale', type=str, default='min_max')
    # parser.add_argument('--dataset_name', type=str, default='NYC')
    parser.add_argument('--thr', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--num_gpu', type=int, default=0)

    parser.add_argument('--temp', type=int, default=8)
    parser.add_argument('--nf', type=int, default=32)
    parser.add_argument('--enf', type=int, default=64)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--es', type=str, default='min')
    parser.add_argument('--input_shape',type=list,default=[8, 8, 4])
    parser.add_argument('--region',type=str,default = 'seoul')

    parser.add_argument('--prediction', type=bool, default=False)

    args = parser.parse_args()
    input_shape = [8, 8,4]

    if args.model_name == 'no_named':
        raise IOError(repr("NO MODEL NAME IN args: python main.py ... --model_name MODEL_NAME"))
    print('[*] Command : python main.py --num_gpu '+str(args.num_gpu)+' --model_name '+str(args.model_name)+' --region '+str(args.region)+' --test --scale ' + (args.scale))
    print('[!] Model Creation Start')
    print('Time is : ', datetime.datetime.now())
    models = TGNet(input_shape,args)
    print('[*] Model Creation End')
    print('Time is : ', datetime.datetime.now())

    if not args.test:
        print('\n [!] Train Start')
        print('Time is : ', datetime.datetime.now())
        models.train(args.region,args.test)
    else:
        if args.prediction:
            print('\n [!] Prediction Start')
            print('Time is : ', datetime.datetime.now())
            models.prediction(args.region)
            print('\n [!] Prediction end')
            print('Time is : ', datetime.datetime.now())
        else:
            print('\n [!] Test Start')
            print('Time is : ', datetime.datetime.now())
            models.test(args.region,args.test)
            print('\n [*] Test End')
            print('Time is : ', datetime.datetime.now())

    print('[!] PROGRAM ENDS')
    print('Time is : ', datetime.datetime.now())