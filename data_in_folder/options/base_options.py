import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):
        parser.add_argument('--rootpath', default='path', type=str, help='Input file path')
        parser.add_argument('--model', default='', type=str, help='Model name')
        parser.add_argument('--output', default='output.json', type=str, help='Output file path')
        parser.add_argument('--mode', default='score', type=str, help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')
        parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
        parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
        parser.add_argument('--model_depth', default=34, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
        parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
        parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')    
        parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
        parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
        parser.set_defaults(verbose=False)
        parser.add_argument('--verbose', action='store_true', help='')
        parser.set_defaults(verbose=False)

        parser.add_argument('--mean', default=[114.7748, 107.7354, 99.4750], type=list, help='Normalization mean')
        parser.add_argument('--std', default=[1, 1, 1], type=list, help='Normalization std')
        parser.add_argument('--device_num', default=0, type=int, help='GPU device number')
        parser.add_argument('--convert', default=True, type=bool, help='Convert to tensor and normalize')
        parser.add_argument('--n_input_channels', default=3, type=int, help='number of input channels') 
        parser.add_argument('--result_path', default='path', type=str, help='path to save the models')    
        parser.add_argument('--conv1_t_size',
                            default=7,
                            type=int,
                            help='Kernel size in t dim of conv1.')    
        parser.add_argument('--conv1_t_stride',
                            default=1,
                            type=int,
                            help='Stride in t dim of conv1.')    
        parser.add_argument('--no_max_pool',
                            action='store_true',
                            help='If true, the max pooling after conv1 is removed.')  
        parser.add_argument(
            '--resnet_widen_factor',
            default=1.0,
            type=float,
            help='The number of feature maps of resnet is multiplied by this value')  
        parser.add_argument('--n_classes', default=4, type=int, help='number of classes')
        
        self.initialized = True
        return parser
    
    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            self.initialize(parser)
        opt, _ = parser.parse_known_args()
        opt.istTrain = self.isTrain
        
        self.opt = parser.parse_args()
        return opt