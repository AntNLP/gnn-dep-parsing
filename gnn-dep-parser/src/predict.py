import argparse, _pickle, math, os, random, sys, time, logging
random.seed(666)
import numpy as np
np.random.seed(666)
from collections import Counter
from antu.io.configurators.ini_configurator import IniConfigurator
from antu.io.vocabulary import Vocabulary
from antu.io.ext_embedding_readers import glove_reader
# from antu.io.datasets.single_task_dataset import DatasetSetting, SingleTaskDataset
from utils.PTB_dataset import DatasetSetting, PTBDataset
from utils.conllu_reader import PTBReader
from antu.utils.dual_channel_logger import dual_channel_logger


def main():
    # Configuration file processing
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../configs/debug.cfg')
    argparser.add_argument('--continue_training', action='store_true', help='Load model Continue Training')
    argparser.add_argument('--name', default='experiment', help='The name of the experiment.')
    argparser.add_argument(
        '--model', default='s2s',
        help='s2s: seq2seq-head-selection-model'
             's2tBFS: seq2tree-BFS-decoder-model'
             's2tDFS: seq2tree-DFS-decoder-model')
    argparser.add_argument('--gpu', default='0', help='GPU ID (-1 to cpu)')
    args, extra_args = argparser.parse_known_args()
    cfg = IniConfigurator(args.config_file, extra_args)

    # Logger setting
    logger = logging.getLogger(__name__) 
    logger.setLevel(logging.DEBUG) 
    ch = logging.StreamHandler() 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m-%d %H:%M')
    ch.setFormatter(formatter) 
    logger.addHandler(ch)
    from eval.ptb_evaluator import PTBEvaluator
    from eval.script_evaluator import ScriptEvaluator

    # DyNet setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import dynet_config
    dynet_config.set(mem=cfg.DYNET_MEM, random_seed=cfg.DYNET_SEED)
    dynet_config.set_gpu()
    import dynet as dy
    from models.token_representation import TokenRepresentation
    from antu.nn.dynet.seq2seq_encoders.rnn_builder import DeepBiLSTMBuilder, orthonormal_VanillaLSTMBuilder
    from models.seq_head_sel_decoder import SeqHeadSelDecoder
    from models.graph_nn_decoder import GraphNNDecoder

    # Build the dataset of the training process
    ## Build data reader
    data_reader = PTBReader(
        field_list=['word', 'tag', 'head', 'rel'],
        root='0\t**root**\t_\t**rpos**\t**rcpos**\t_\t0\t**rrel**\t_\t_',
        spacer=r'[\t]',)
    ## Build vocabulary with pretrained glove
    vocabulary = Vocabulary()
    g_word, _ = glove_reader(cfg.GLOVE)
    pretrained_vocabs = {'glove': g_word}
    vocabulary.extend_from_pretrained_vocab(pretrained_vocabs)
    ## Setup datasets
    datasets_settings = {
        'train': DatasetSetting(cfg.TRAIN, True),
        'dev': DatasetSetting(cfg.DEV, False),
        'test': DatasetSetting(cfg.TEST, False),}
    datasets = PTBDataset(vocabulary, datasets_settings, data_reader)
    counters = {'word': Counter(), 'tag': Counter(), 'rel': Counter()}
    datasets.build_dataset(
        counters, no_pad_namespace={'rel'}, no_unk_namespace={'rel'})

    my_eval = ScriptEvaluator(['Valid', 'Test'], datasets.vocabulary)

    test_pc = dy.ParameterCollection()
    token_repre, encoder, decoder = dy.load(cfg.BEST_FILE, test_pc)
    my_eval.clear('Test')
    def cmp(ins): return len(ins['word'])
    t1 = time.time()
    test_batch = datasets.get_batches('test', cfg.TEST_BATCH_SIZE, False, cmp, False)
    for indexes, masks, truth in test_batch:
        dy.renew_cg()
        vectors = token_repre(indexes, False)
        vectors = encoder(vectors, None, cfg.RNN_X_DROP, cfg.RNN_H_DROP, False)
        pred = decoder(vectors, masks, None, False, True)
        my_eval.add_truth('Test', truth)
        my_eval.add_pred('Test', pred)
    t2 = time.time()
    logger.info('Test time: %f' % (t2-t1))
    my_eval.evaluation('Test', cfg.PRED_TEST, cfg.TEST)


if __name__ == '__main__':
    main()