from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        parser.add_argument('--no_val', default=False, type=bool, help='existance of validation set')
        parser.add_argument('--no_train', default=False, type=bool, help='existance of training set')        
        parser.add_argument('--begin_epoch', default=0, type=int, help='begin epoch')
        parser.add_argument('--n_epochs', default=1000, type=int, help='number of epochs')
        
        parser.add_argument('--checkpoint',
                            default=10,
                            type=int,
                            help='Trained model is saved at every this epochs.')  
        parser.add_argument('--experiment', default="experiment", type=str, help='the name of the experiment')
        parser.add_argument('--tensorboard',
                            action='store_true',
                            help='If true, output tensorboard log file.')            
        self.isTrain = True
        return parser