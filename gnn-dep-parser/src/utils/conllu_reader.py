from typing import Callable, List, Dict
from overrides import overrides
import re, sys
from collections import Counter
from antu.io.instance import Instance
from antu.io.fields.field import Field
from antu.io.fields.text_field import TextField
from antu.io.fields.index_field import IndexField
from antu.io.token_indexers.token_indexer import TokenIndexer
from antu.io.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from antu.io.dataset_readers.dataset_reader import DatasetReader


class PTBReader(DatasetReader):

    def __init__(
        self,
        field_list: List[str],
        root: str,
        spacer: str):

        self.field_list = field_list
        self.root = root
        self.spacer = spacer

    def _read(self, file_path: str) -> Instance:
        with open(file_path, 'rt') as fp:
            root_token = re.split(self.spacer, self.root)
            tokens = [[item,] for item in root_token]
            for line in fp:
                token = re.split(self.spacer, line.strip())
                if line.strip() == '':
                    if len(tokens[0]) > 1: yield tokens
                    tokens = [[item,] for item in root_token]
                else:
                    for idx, item in enumerate(token):
                        tokens[idx].append(item)
            if len(tokens[0]) > 1: yield tokens

    @overrides
    def read(self, file_path: str) -> List[Instance]:
        # Build indexers
        indexers = dict()
        word_indexer = SingleIdTokenIndexer(
            ['word', 'glove'], (lambda x:x.casefold()))
        indexers['word'] = [word_indexer,]
        tag_indexer = SingleIdTokenIndexer(['tag'])
        indexers['tag'] = [tag_indexer,]
        rel_indexer = SingleIdTokenIndexer(['rel'])
        indexers['rel'] = [rel_indexer,]

        # Build instance list
        res = []
        for sentence in self._read(file_path):
            res.append(self.input_to_instance(sentence, indexers))
        return res

    @overrides
    def input_to_instance(
        self,
        inputs: List[List[str]],
        indexers: Dict[str, List[TokenIndexer]]) -> Instance:
        fields = []
        if 'word' in self.field_list:
            fields.append(TextField('word', inputs[1], indexers['word']))
        if 'tag' in self.field_list:
            fields.append(TextField('tag', inputs[4], indexers['tag']))
        if 'head' in self.field_list:
            fields.append(IndexField('head', inputs[6]))
        if 'rel' in self.field_list:
            fields.append(TextField('rel', inputs[7], indexers['rel']))
        return Instance(fields)