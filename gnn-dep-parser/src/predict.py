from antu.utils.dual_channel_logger import dual_channel_logger
from utils.conllu_reader import PTBReader
from utils.PTB_dataset import DatasetSetting, PTBDataset
from antu.io.ext_embedding_readers import glove_reader
from antu.io import Vocabulary
from antu.io.configurators import IniConfigurator
from collections import Counter
import numpy as np
import argparse
import _pickle
import math
import os
import random
import sys
import time
import logging
random.seed(666)
np.random.seed(666)


def main():
    # Configuration file processing
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/debug.cfg')
    argparser.add_argument('--continue_training', action='store_true',
                           help='Load model Continue Training')
    argparser.add_argument('--name', default='experiment',
                           help='The name of the experiment.')
    argparser.add_argument('--model', default='s2s',
                           help='s2s: seq2seq-head-selection-model'
                           's2tBFS: seq2tree-BFS-decoder-model'
                           's2tDFS: seq2tree-DFS-decoder-model')
    argparser.add_argument('--gpu', default='0', help='GPU ID (-1 to cpu)')
    args, extra_args = argparser.parse_known_args()
    cfg = IniConfigurator(args.config_file, extra_args)

    # Logger setting
    logger = dual_channel_logger(
        __name__,
        file_path=cfg.LOG_FILE,
        file_model='w',
        formatter='%(asctime)s - %(levelname)s - %(message)s',
        time_formatter='%m-%d %H:%M')
    from eval.script_evaluator import ScriptEvaluator

    # DyNet setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import dynet_config
    dynet_config.set(mem=cfg.DYNET_MEM, random_seed=cfg.DYNET_SEED)
    dynet_config.set_gpu()
    import dynet as dy

    # Build the dataset of the training process
    # Build data reader
    data_reader = PTBReader(
        field_list=['word', 'tag', 'head', 'rel'],
        root='0\t**root**\t_\t**rcpos**\t**rpos**\t_\t0\t**rrel**\t_\t_',
        spacer=r'[\t]',)
    # Build vocabulary with pretrained glove
    vocabulary = Vocabulary()
    g_word, _ = glove_reader(cfg.GLOVE)
    pretrained_vocabs = {'glove': g_word}
    vocabulary.extend_from_pretrained_vocab(pretrained_vocabs)
    # Setup datasets
    datasets_settings = {'train': DatasetSetting(cfg.TRAIN, True),
                         'dev': DatasetSetting(cfg.DEV, False),
                         'test': DatasetSetting(cfg.TEST, False), }
    datasets = PTBDataset(vocabulary, datasets_settings, data_reader)
    counters = {'word': Counter(), 'tag': Counter(), 'rel': Counter()}
    datasets.build_dataset(counters, no_pad_namespace={'rel'},
                           no_unk_namespace={'rel'})

    logger.info("Experiment name: %s" % args.name)
    SHA = os.popen('git log -1 | head -n 1 | cut -c 8-13').readline().rstrip()
    logger.info('Git SHA: %s' % SHA)

    # Build Test model
    test_pc = dy.ParameterCollection()
    token_repre, encoder, decoder = dy.load(cfg.BEST_FILE, test_pc)

    # PTB Evaluator
    my_eval = ScriptEvaluator(['Valid', 'Test'], datasets.vocabulary)
    my_eval.clear('Test')

    def cmp(ins):
        return len(ins['word'])
    test_batch = datasets.get_batches('test', cfg.TEST_BATCH_SIZE, False, cmp,
                                      False)
    for indexes, masks, truth in test_batch:
        dy.renew_cg()
        vectors = token_repre(indexes, False)
        vectors = encoder(vectors, None, cfg.RNN_DROP, cfg.RNN_DROP, False)
        pred = decoder(vectors, masks, None, False, True)
        my_eval.add_truth('Test', truth)
        my_eval.add_pred('Test', pred)
    my_eval.evaluation('Test', cfg.PRED_TEST, cfg.TEST)


if __name__ == '__main__':
    main()
