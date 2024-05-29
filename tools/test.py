import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
parent_dir_path2 = os.path.abspath(os.path.join(parent_dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
sys.path.insert(0, parent_dir_path2)


from dslstt.tools.demo_eval import get_parser
from dslstt.networks.managers.evaluator import EvaluatorVisha


if __name__ == '__main__':
    cfg = get_parser()
    evaluator = EvaluatorVisha(cfg)
    evaluator.evaluating()