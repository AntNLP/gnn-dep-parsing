from typing import Dict, TypeVar, List
from antu.io.vocabulary import Vocabulary
from antu.io.ext_embedding_readers import glove_reader
Indices = TypeVar("Indices", List[int], List[List[int]])
import numpy as np
import dynet as dy
import random


class TokenRepresentation(object):

    def __init__(
        self,
        model,
        cfg,
        vocabulary: Vocabulary):

        pc = model.add_subcollection()
        word_num = vocabulary.get_vocab_size('word')
        self.wlookup = pc.lookup_parameters_from_numpy(
            np.zeros((word_num, cfg.WORD_DIM), dtype=np.float32))
        tag_num = vocabulary.get_vocab_size('tag')
        self.tlookup = pc.lookup_parameters_from_numpy(
            np.random.randn(tag_num, cfg.TAG_DIM).astype(np.float32))
        _, glove_vec = glove_reader(cfg.GLOVE)
        glove_dim = len(glove_vec[0])
        unk_pad_vec = [[0.0 for _ in range(glove_dim)]]
        glove_num = vocabulary.get_vocab_size('glove')
        glove_vec = unk_pad_vec + unk_pad_vec + glove_vec
        glove_vec = np.array(glove_vec, dtype=np.float32)/np.std(glove_vec)
        self.glookup = pc.lookup_parameters_from_numpy(glove_vec.astype(np.float32))

        self.token_dim = cfg.WORD_DIM + cfg.TAG_DIM
        self.vocabulary = vocabulary
        self.pc, self.cfg = pc, cfg
        self.spec = (cfg, vocabulary)

    def __call__(
        self,
        indexes: Dict[str, List[Indices]],
        is_train=False) -> List[dy.Expression]:
        len_s = len(indexes['head'][0])
        batch_num = len(indexes['head'])
        vectors = []
        for i in range(len_s):
            # map token indexes -> vector
            w_idxes = [indexes['word']['word'][x][i] for x in range(batch_num)]
            g_idxes = [indexes['word']['glove'][x][i] for x in range(batch_num)]
            t_idxes = [indexes['tag']['tag'][x][i] for x in range(batch_num)]
            w_vec = dy.lookup_batch(self.wlookup, w_idxes)
            g_vec = dy.lookup_batch(self.glookup, g_idxes, False)
            w_vec += g_vec
            t_vec = dy.lookup_batch(self.tlookup, t_idxes)

            # build token mask with dropout scale
            # For only word dropped: tag * 3
            # For only tag dropped: word * 1.5
            # For both word and tag dropped: 0 vector
            if is_train:
                wm = np.random.binomial(1, 1.-self.cfg.WORD_DROP, batch_num).astype(np.float32)
                tm = np.random.binomial(1, 1.-self.cfg.TAG_DROP, batch_num).astype(np.float32)
                scale = np.logical_or(wm, tm) * 3 / (2*wm + tm + 1e-12)
                wm *= scale
                tm *= scale
                w_vec *= dy.inputTensor(wm, batched=True)
                t_vec *= dy.inputTensor(tm, batched=True)
            vectors.append(dy.concatenate([w_vec, t_vec]))
        return vectors

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        cfg, vocabulary = spec
        return TokenRepresentation(model, cfg, vocabulary)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc
