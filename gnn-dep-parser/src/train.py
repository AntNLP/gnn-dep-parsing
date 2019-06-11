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
    logger = dual_channel_logger(
        __name__,
        file_path=cfg.LOG_FILE,
        file_model='w',
        formatter='%(asctime)s - %(levelname)s - %(message)s',
        time_formatter='%m-%d %H:%M')
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
        root='0\t**root**\t_\t**rcpos**\t**rpos**\t_\t0\t**rrel**\t_\t_',
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

    # Build model
    pc = dy.ParameterCollection()
    trainer = dy.AdamTrainer(
        pc,
        alpha=cfg.LEARNING_RATE,
        beta_1=cfg.ADAM_BETA1,
        beta_2=cfg.ADAM_BETA2,
        eps=cfg.EPS)
    BEST_DEV_LAS = BEST_DEV_UAS = 0

    token_repre = TokenRepresentation(pc, cfg, datasets.vocabulary)
    encoder = DeepBiLSTMBuilder(
        pc,
        cfg.ENC_LAYERS, token_repre.token_dim, cfg.ENC_H_DIM,
        orthonormal_VanillaLSTMBuilder, param_init=True, fb_fusion=True)
    decoder = GraphNNDecoder(
        pc,
        cfg,
        datasets.vocabulary)
    # Train model
    cnt_iter = 0
    def cmp(ins): return len(ins['word'])
    train_batch = datasets.get_batches('train', cfg.TRAIN_BATCH_SIZE, True, cmp, True)

    my_eval = ScriptEvaluator(['Valid', 'Test'], datasets.vocabulary)
    valid_loss = [[] for i in range(cfg.GRAPH_LAYERS+3)] 
    logger.info("Experiment name: %s" % args.name)
    logger.info('Git SHA: %s' % os.popen('git log -1 | head -n 1 | cut -c 8-13').readline().rstrip())
    while cnt_iter < cfg.MAX_ITER:
        dy.renew_cg()
        cnt_iter += 1
        indexes, masks, truth = train_batch.__next__()
        #print(indexes)
        vectors = token_repre(indexes, True)
        vectors = encoder(vectors, None, cfg.RNN_X_DROP, cfg.RNN_H_DROP, True)
        loss, part_loss = decoder(vectors, masks, truth, True, True)
        for i, l in enumerate([loss]+part_loss):
            valid_loss[i].append(l.value())
        loss.backward()
        trainer.learning_rate = cfg.LEARNING_RATE*cfg.LR_DECAY**(cnt_iter / cfg.LR_ANNEAL)
        trainer.update()

        if cnt_iter % cfg.VALID_ITER: continue
        for i in range(len(valid_loss)): valid_loss[i] = str(np.mean(valid_loss[i]))
        avg_loss = ', '.join(valid_loss)
        logger.info("")
        logger.info("Iter: %d-%d, Avg_loss: %s" % (cnt_iter / cfg.VALID_ITER, cnt_iter, avg_loss))
        valid_loss = [[] for i in range(cfg.GRAPH_LAYERS+3)] 
        my_eval.clear('Valid')
        valid_batch = datasets.get_batches('dev', cfg.TEST_BATCH_SIZE, False, cmp, False)
        for indexes, masks, truth in valid_batch:
            dy.renew_cg()
            vectors = token_repre(indexes, False)
            vectors = encoder(vectors, None, cfg.RNN_X_DROP, cfg.RNN_H_DROP, False)
            pred = decoder(vectors, masks, None, False, True)
            my_eval.add_truth('Valid', truth)
            my_eval.add_pred('Valid', pred)
        dy.save(cfg.LAST_FILE, [token_repre, encoder, decoder])
        if my_eval.evaluation('Valid', cfg.PRED_DEV, cfg.DEV):
            os.system('cp %s.data %s.data' % (cfg.LAST_FILE, cfg.BEST_FILE))
            os.system('cp %s.meta %s.meta' % (cfg.LAST_FILE, cfg.BEST_FILE))
        my_eval.clear('Test')
        test_batch = datasets.get_batches('test', cfg.TEST_BATCH_SIZE, False, cmp, False)
        for indexes, masks, truth in test_batch:
            dy.renew_cg()
            vectors = token_repre(indexes, False)
            vectors = encoder(vectors, None, cfg.RNN_X_DROP, cfg.RNN_H_DROP, False)
            pred = decoder(vectors, masks, None, False, True)
            my_eval.add_truth('Test', truth)
            my_eval.add_pred('Test', pred)
        my_eval.evaluation('Test', cfg.PRED_TEST, cfg.TEST)
    my_eval.print_best_result('Valid')
    test_pc = dy.ParameterCollection()
    token_repre, encoder, decoder = dy.load(cfg.BEST_FILE, test_pc)
    my_eval.clear('Test')
    test_batch = datasets.get_batches('test', cfg.TEST_BATCH_SIZE, False, cmp, False)
    for indexes, masks, truth in test_batch:
        dy.renew_cg()
        vectors = token_repre(indexes, False)
        vectors = encoder(vectors, None, cfg.RNN_X_DROP, cfg.RNN_H_DROP, False)
        pred = decoder(vectors, masks, None, False, True)
        my_eval.add_truth('Test', truth)
        my_eval.add_pred('Test', pred)
    my_eval.evaluation('Test', cfg.PRED_TEST, cfg.TEST)


if __name__ == '__main__':
    main()