# options/train_options.py

import argparse
from .base_options import Options

class TrainOptions(Options):
    """Class to handle all training options and arguments."""

    def __init__(self):
        super(TrainOptions, self).__init__()
        self.initialize()

    def initialize(self):
        super(TrainOptions, self).initialize()
        # Model architecture
        self.parser.add_argument('--num_res_blocks', type=int, default=2, help='Number of residual blocks in encoder/decoder.')
        self.parser.add_argument('--use_attention', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use self-attention in encoder/decoder.')
        self.parser.add_argument('--use_embedding', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use embedding for categorical variables.')
        self.parser.add_argument('--use_batch_norm', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use batch normalization.')
        self.parser.add_argument('--use_dropout', type=lambda x: (str(x).lower() == 'true'), default=True, help='Use dropout in encoder/decoder.')

        # Training hyperparameters
        self.parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
        self.parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
        self.parser.add_argument('--learning_rate', type=float, default=0.0002, help='Initial learning rate.')
        self.parser.add_argument('--betas', type=float, nargs=2, default=(0.5, 0.9), help='Betas for Adam optimizer.')
        self.parser.add_argument('--clip_value', type=float, default=0.01, help='Gradient clipping value.')

        # Loss weights
        self.parser.add_argument('--lambda_kl', type=float, default=0.01, help='Weight for KL divergence.')
        self.parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Weight for cycle consistency loss.')
        self.parser.add_argument('--lambda_gan', type=float, default=1.0, help='Weight for GAN loss.')

        # Early stopping
        self.parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping.')

        # Resume training
        self.parser.add_argument('--load_model', action='store_true', help='Load model from checkpoint if available.')
        self.parser.add_argument('--checkpoint_path', type=str, default='', help='Path to checkpoint file.')

    
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        # Print all options
        print('------------ Training Options -------------')
        for k, v in sorted(vars(self.opt).items()):
            print(f'{k}: {v}')
        print('------------------ End --------------------')
        return self.opt