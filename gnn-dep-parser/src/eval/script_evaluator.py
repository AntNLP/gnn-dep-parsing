import logging
logger = logging.getLogger('__main__')

class ScriptEvaluator:

    def __init__(self, names, vocab):
        self.eval_set = {}
        for name in names:
            self.eval_set[name] = {}
        self.ignore_tag = set()
        self.ignore_tag.add(vocab.get_token_index('``', 'tag'))
        self.ignore_tag.add(vocab.get_token_index('\'\'', 'tag'))
        self.ignore_tag.add(vocab.get_token_index(':', 'tag'))
        self.ignore_tag.add(vocab.get_token_index(',', 'tag'))
        self.ignore_tag.add(vocab.get_token_index('.', 'tag'))
        self.root_idx = vocab.get_token_index('root', 'rel')
        self.vocab = vocab

    def add_pred(self, name, pred):
        self.eval_set[name]['pred'].append(pred)

    def add_truth(self, name, truth):
        self.eval_set[name]['truth'].append(truth)

    def evaluation(self, name, pred_path=None, gold_path=None):
        if 'Best_UAS' not in self.eval_set[name]:
            self.eval_set[name]['Best_UAS'] = 0
            self.eval_set[name]['Best_LAS'] = 0
        total_token = UA = LA = cnt = 0
        if pred_path:
            fout = open(pred_path, 'w')
        pred_list = []
        for truth, pred in zip(self.eval_set[name]['truth'], self.eval_set[name]['pred']):
            for truth_head, pred_head, truth_rel, pred_rel, tag in zip(truth['head'], pred['head'], truth['rel'], pred['rel'], truth['tag']):
                if pred_head == 0: pred_rel = self.root_idx
                if (truth_head or truth_rel) and tag not in self.ignore_tag:
                    total_token += 1
                    if truth_head == pred_head:
                        UA += 1
                        if truth_rel == pred_rel:
                            LA += 1
                if (truth_head or truth_rel):
                    rel_str = self.vocab.get_token_from_index(pred_rel, 'rel')
                    pred_list.append((str(pred_head), rel_str))
                    cnt += 1
        with open(gold_path, 'r') as fin:
            lines = fin.readlines()
            j = 0
            for line in lines:
                if line.strip() == '':
                    fout.write(line)
                else:
                    ins = line.strip().split('\t')
                    ins[6] = pred_list[j][0]
                    ins[7] = pred_list[j][1]
                    pred_line = '\t'.join(ins) + '\n'
                    fout.write(pred_line)
                    j += 1
        fout.close()

        UAS = UA*1.0/total_token
        LAS = LA*1.0/total_token
        logger.info("%s: UAS=%f, LAS=%f" % (name, UAS, LAS))
        if UAS > self.eval_set[name]['Best_UAS'] or (UAS+LAS > self.eval_set[name]['Best_UAS']+self.eval_set[name]['Best_LAS']):
            self.eval_set[name]['Best_UAS'] = UAS
            self.eval_set[name]['Best_LAS'] = LAS
            return True
        return False

    def print_best_result(self, name):
        logger.info('Best Results: UAS: %s, LAS: %s' % (self.eval_set[name]['Best_UAS'], self.eval_set[name]['Best_LAS']))

    def clear(self, name):
        self.eval_set[name]['pred'] = []
        self.eval_set[name]['truth'] = []