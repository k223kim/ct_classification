import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvpath', default='path', type=str, help='csv file path')
    parser.add_argument('--model', default='', type=str, help='Model name')
    parser.add_argument('--output', default='output.json', type=str, help='Output file path')
    parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--model_depth', default=34, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')    
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(verbose=False)
    parser.add_argument('--verbose', action='store_true', help='')
    parser.set_defaults(verbose=False)

    parser.add_argument('--mean', default=[0.5], type=list, help='Normalization mean')
    parser.add_argument('--std', default=[0.5], type=list, help='Normalization std')
    parser.add_argument('--device_num', default=0, type=int, help='GPU device number')
    parser.add_argument('--convert', default=True, type=bool, help='Convert to tensor and normalize')
    parser.add_argument('--no_val', default=False, type=bool, help='existance of validation set')
    parser.add_argument('--no_train', default=False, type=bool, help='existance of training set')
    parser.add_argument('--n_input_channels', default=3, type=int, help='number of input channels')
    parser.add_argument('--checkpoint',
                        default=10,
                        type=int,
                        help='Trained model is saved at every this epochs.')   
    parser.add_argument('--result_path', default='path', type=str, help='path to save the models')    
    
    parser.add_argument('--n_classes', default=4, type=int, help='number of classes')
    args = parser.parse_args()

    return args