# options/base_options.py

import argparse

class Options:
    """Base class for all option classes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Base Options for CycSeq")
        self.initialized = False

    def initialize(self):
        # 通用参数
        self.parser.add_argument('--name', type=str, default='cycseq', help='Experiment name.')
        self.parser.add_argument('--model_output_path', type=str, default='./output_model', help='Model output directory.')
        self.parser.add_argument('--log_dir', type=str, default='./logs', help='TensorBoard log directory.')
        self.parser.add_argument('--feature_path', type=str, default='', help='Path to feature .npy file.')
        self.parser.add_argument('--label_path', type=str, default='', help='Path to label .npy file.')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        print('------------ Options -------------')
        for k, v in sorted(vars(self.opt).items()):
            print(f'{k}: {v}')
        print('-------------- End ----------------')
        return self.opt