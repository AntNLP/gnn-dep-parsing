from typing import Dict, List, Callable, Union, Set
from overrides import overrides
from antu.io.vocabulary import Vocabulary
from antu.io.instance import Instance
from antu.io.datasets.dataset import Dataset
from antu.io.dataset_readers.dataset_reader import DatasetReader
import random
from itertools import cycle
import numpy as np


class DatasetSetting:

    def __init__(self, file_path: str, is_train: bool):
        self.file_path = file_path
        self.is_train = is_train



def shadow_padding(batch_input, vocabulary):
    maxlen = 0
    for ins in batch_input:
        maxlen = max(maxlen, len(ins['head']))

    masks = {'1D': list()}
    truth = {'word': [], 'tag': [], 'head': [], 'flat_head': [], 'rel': []}
    inputs = {
        'word': {'word': [], 'glove': []},
        'tag': {'tag': []},
        'head': [],
        'rel': {'rel': []},
    }
    for ins in batch_input:
        padding_length = maxlen - len(ins['head'])
        # PAD word
        padding_index = vocabulary.get_padding_index('word')
        pad_seq = [padding_index] * padding_length
        inputs['word']['word'].append(ins['word']['word']+pad_seq)
        truth['word'].extend(inputs['word']['word'][-1])
        # PAD glove
        padding_index = vocabulary.get_padding_index('glove')
        pad_seq = [padding_index] * padding_length
        inputs['word']['glove'].append(ins['word']['glove']+pad_seq)
        # PAD tag
        padding_index = vocabulary.get_padding_index('tag')
        pad_seq = [padding_index] * padding_length
        inputs['tag']['tag'].append(ins['tag']['tag']+pad_seq)
        truth['tag'].extend(inputs['tag']['tag'][-1])
        # PAD head
        padding_index = 0
        pad_seq = [padding_index] * padding_length
        inputs['head'].append(ins['head']+pad_seq)
        flat_head = (len(truth['head'])+np.array(inputs['head'][-1])).tolist()
        truth['flat_head'].extend(flat_head)
        truth['head'].extend(inputs['head'][-1])
        # PAD rel
        padding_index = 0
        pad_seq = [padding_index] * padding_length
        inputs['rel']['rel'].append(ins['rel']['rel']+pad_seq)
        truth['rel'].extend(inputs['rel']['rel'][-1])
        # Mask
        ins_mask = [1]*(maxlen-padding_length) + [0]*padding_length
        masks['1D'].append(ins_mask)

    # Build [1D], [2D], [Flat] masks
    zero = [0] * maxlen
    masks['2D'] = []    # batch_size * sent_len
    masks['flat'] = []
    for ins in masks['1D']:
        no_pad = sum(ins)
        masks['2D'].append([ins] * no_pad)
        masks['2D'][-1].extend([zero] * (maxlen-no_pad))
        ins[0] = 0
        masks['flat'].extend(ins)
        ins[0] = 1
    masks['2D'] = np.transpose(np.array(masks['2D'])) - np.eye(maxlen).reshape((maxlen, maxlen, 1))
    masks['2D'][masks['2D'] == -1] = 0
    masks['flat'] = np.array(masks['flat'])
    return inputs, masks, truth


class PTBDataset:

    def __init__(
        self,
        vocabulary: Vocabulary,
        datasets_settings: Dict[str, DatasetSetting],
        reader: DatasetReader):
        self.vocabulary = vocabulary
        self.datasets_settings = datasets_settings
        self.datasets = dict()
        self.reader = reader
        self.is_padding = dict()
        for name, setting in self.datasets_settings.items():
            if not setting.is_train:
                self.is_padding[name] = False
        self.paddataset = {}

    def build_dataset(
        self,
        counters: Dict[str, Dict[str, int]],
        min_count: Union[int, Dict[str, int]] = dict(),
        no_pad_namespace: Set[str] = set(),
        no_unk_namespace: Set[str] = set()) -> None:

        for name, setting in self.datasets_settings.items():
            self.datasets[name] = self.reader.read(setting.file_path)
            if setting.is_train:
                for ins in self.datasets[name]:
                    ins.count_vocab_items(counters)
        self.vocabulary.extend_from_counter(
            counters, min_count, no_pad_namespace, no_unk_namespace)
        for name in self.datasets:
            for ins in self.datasets[name]:
                ins.index_fields(self.vocabulary)

    def get_dataset(self, name: str) -> List[Instance]:
        return self.datasets[name]

    def get_batches(
        self,
        name: str,
        size: int,
        ordered: bool=False,
        cmp: Callable[[Instance, Instance], int]=None,
        is_infinite: bool=False) -> List[List[int]]:
        #print(self.datasets[name])
        if ordered: self.datasets[name].sort(key=cmp)

        num = len(self.datasets[name]) # Number of Instances
        result = []
        if is_infinite:
            for beg in range(0, num, size):
                ins_batch = self.datasets[name][beg: beg+size]
                idx_batch = [ins.index_fields(self.vocabulary) for ins in ins_batch]
                indexes, masks, truth= shadow_padding(idx_batch, self.vocabulary)
                yield indexes, masks, truth
                result.append((indexes, masks, truth))
        else:
            if not self.is_padding[name]:
                for beg in range(0, num, size):
                    ins_batch = self.datasets[name][beg: beg+size]
                    idx_batch = [ins.index_fields(self.vocabulary) for ins in ins_batch]
                    indexes, masks, truth= shadow_padding(idx_batch, self.vocabulary)
                    result.append((indexes, masks, truth))
                self.is_padding[name] = True
                self.paddataset[name] = result
            for indexes, masks, truth in self.paddataset[name]:
                yield indexes, masks, truth

        while is_infinite:
            random.shuffle(result)
            for indexes, masks, truth in result:
                yield indexes, masks, truth

    # def build_batches(self, )
