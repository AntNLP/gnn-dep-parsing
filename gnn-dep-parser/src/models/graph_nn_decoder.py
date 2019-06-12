from .dep_decoder import DependencyDecoder
from antu.io.vocabulary import Vocabulary
from antu.io.configurators.ini_configurator import IniConfigurator
from antu.nn.dynet.multi_layer_perception import MLP
from antu.nn.dynet.attention.biaffine import BiaffineAttention
from antu.nn.dynet.units.graph_nn_unit import GraphNNUnit
from utils.mst_decoder import MST_inference
import dynet as dy
import numpy as np


def leaky_relu(x):
    return dy.bmax(.1*x, x)


class GraphNNDecoder(DependencyDecoder):

    def __init__(self,
                 model: dy.ParameterCollection,
                 cfg: IniConfigurator,
                 vocabulary: Vocabulary):

        pc = model.add_subcollection()
        # MLP layer
        self.head_MLP = MLP(pc, cfg.MLP_SIZE, leaky_relu, 'orthonormal',
                            cfg.MLP_BIAS, cfg.MLP_DROP)
        self.dept_MLP = MLP(pc, cfg.MLP_SIZE, leaky_relu, 'orthonormal',
                            cfg.MLP_BIAS, cfg.MLP_DROP)

        # Biaffine Attention Layer (Arc)
        arc_size = cfg.ARC_SIZE
        self.arc_attn_mat = [
            BiaffineAttention(pc, arc_size, arc_size, 1, cfg.ARC_BIAS, 0)
            for _ in range(cfg.GRAPH_LAYERS+1)]

        # Biaffine Attention Layer (Rel)
        rel_num = vocabulary.get_vocab_size('rel')
        rel_size = cfg.MLP_SIZE[-1]-cfg.ARC_SIZE
        self.rel_mask = np.array([1] + [0] * (rel_num-1))   # mask root relation
        self.rel_attn = BiaffineAttention(pc, rel_size, rel_size, rel_num,
                                          cfg.REL_BIAS, 0)

        # Graph Network Layer
        self.head_gnn = GraphNNUnit(pc, arc_size, arc_size, leaky_relu,
                                    'orthonormal')
        self.dept_gnn = GraphNNUnit(pc, arc_size, arc_size, leaky_relu,
                                    'orthonormal')

        # Save Variable
        self.arc_size, self.rel_size, self.rel_num = arc_size, rel_size, rel_num
        self.pc, self.cfg = pc, cfg
        self.spec = (cfg, vocabulary)

    def __call__(self, inputs, masks, truth, is_train=True, is_tree=True):
        sent_len = len(inputs)
        batch_size = inputs[0].dim()[1]
        flat_len = sent_len * batch_size

        # H -> hidden size, L -> sentence length, B -> batch size
        # ((H, L), B)
        X = dy.concatenate_cols(inputs)
        if is_train:
            X = dy.dropout_dim(X, 1, self.cfg.MLP_DROP)
        # M_H -> MLP hidden size
        # ((M_H, L), B)
        head_mat = self.head_MLP(X, is_train)
        # ((M_H, L), B)
        dept_mat = self.dept_MLP(X, is_train)
        if is_train:
            total_token = sum(masks['flat'].tolist())
            head_mat = dy.dropout_dim(head_mat, 1, self.cfg.MLP_DROP)
            dept_mat = dy.dropout_dim(dept_mat, 1, self.cfg.MLP_DROP)

        # A_H -> Arc hidden size, R_H -> Label hidden size, A_H + R_H = M_H
        head_arc = head_mat[:self.arc_size]     # ((A_H, L), B)
        dept_arc = dept_mat[:self.arc_size]     # ((A_H, L), B)
        head_rel = head_mat[self.arc_size:]     # ((R_H, L), B)
        dept_rel = dept_mat[self.arc_size:]     # ((R_H, L), B)

        # ((L, L), B)
        masks_2D = dy.inputTensor(masks['2D'], True)
        # (1, L*B)
        masks_flat = dy.inputTensor(masks['flat'], True)

        gnn_losses = []
        for k in range(self.cfg.GRAPH_LAYERS):
            # Graph Weights
            # ((L, L), B)
            arc_mat  = self.arc_attn_mat[k](head_arc, dept_arc)-1e9*(1-masks_2D)
            arc_prob = dy.softmax(arc_mat)

            # Layer-wise Loss
            if is_train:
                # ((L,), L*B)
                arc_mat = dy.reshape(arc_mat, (sent_len,), flat_len)
                # ((1,), L*B)
                arc_loss = dy.pickneglogsoftmax_batch(arc_mat, truth['head'])
                # (1,)
                arc_loss = dy.sum_batches(arc_loss*masks_flat)/total_token
                gnn_losses.append(arc_loss)

            # Aggregation Function
            # Fusion head and dept representation
            # ((A_H, L), B)
            HX = head_arc * arc_prob
            DX = dept_arc * dy.transpose(arc_prob)
            FX = HX + DX

            # Async Update Function
            # Head-first
            # ((A_H, L), B)
            head_arc = self.head_graphNN(FX, head_arc, is_train)
            FX_new = head_arc * arc_prob + DX
            dept_arc = self.dept_graphNN(FX_new, dept_arc, is_train)

        # ((L, L), B)
        arc_mat = self.arc_attn_mat[-1](head_arc, dept_arc)-1e9*(1-masks_2D)
        # ((L,), L*B)
        arc_mat = dy.reshape(arc_mat, (sent_len,), flat_len)
        # Predict Relation
        # (R_H, L*B)
        head_rel = dy.reshape(head_rel, (self.rel_size, flat_len))
        # ((R_H,), L*B)
        dept_rel = dy.reshape(dept_rel, (self.rel_size,), flat_len)
        if is_train:
            # ((1,), L*B)
            arc_losses = dy.pickneglogsoftmax_batch(arc_mat, truth['head'])
            # (1,)
            arc_loss = dy.sum_batches(arc_losses*masks_flat)/total_token
            # ((R_H,), L*B)
            truth_rel = dy.pick_batch(head_rel, truth['flat_head'], 1)
            # R -> Relation Set Size
            # ((R,), L*B)
            rel_mat = self.rel_attn(son_rel, truth_rel)
        else:
            if is_tree:
                # MST Inference, Achieve Tree Edge.
                arc_probs = dy.softmax(arc_mat).npvalue()
                arc_probs = np.reshape(arc_probs,
                                       (sent_len, sent_len, batch_size), 'F')
                arc_probs = np.transpose(arc_probs)
                # Mask PAD
                arc_masks = [np.array(masks['flat'][i:i+sent_len])
                             for i in range(0, flat_len, sent_len)]
                arc_pred = []
                # Inference One By One.
                for msk, arc_prob in zip(arc_masks, arc_probs):
                    msk[0] = 1
                    seq_len = int(np.sum(msk))
                    tmp_pred = MST_inference(arc_prob, seq_len, msk)
                    tmp_pred[0] = 0
                    arc_pred.extend(tmp_pred)
            else:
                # Greedy Inference (argmax)
                arc_pred = np.argmax(arc_mat.npvalue(), 0)
            # Pick Predicted Edge's <Head, Dept> pair.
            flat_pred = [j+(i//sent_len)*sent_len
                         for i, j in enumerate(arc_pred)]
            pred_rel = dy.pick_batch(head_rel, flat_pred, 1)
            # Predict Relation (mask ROOT)
            rel_mat = self.rel_attn(son_rel, pred_rel)
            rel_mask = dy.inputTensor(self.rel_mask)
            rel_mat = rel_mat - 1e9*rel_mask
        if is_train:
            # Calculate Relation Classification Loss
            # ((1,), L*B)
            rel_losses = dy.pickneglogsoftmax_batch(rel_mat, truth['rel'])
            # (1,)
            rel_loss = dy.sum_batches(rel_losses*masks_flat) / total_token
            # Final Total Loss with Layer-wise
            losses = (rel_loss+arc_loss)*self.cfg.LAMBDA2
            if gnn_losses:
                losses += dy.esum(gnn_losses)*self.cfg.LAMBDA1
            losses_list = gnn_losses + [arc_loss, rel_loss]
            return losses, losses_list
        else:
            rel_mat = dy.reshape(rel_mat, (self.rel_num,)).npvalue()
            rel_pred = np.argmax(rel_mat, 0)
            pred = {}
            pred['head'], pred['rel'] = arc_pred, rel_pred
            return pred

    @staticmethod
    def from_spec(spec, model):
        """Create and return a new instane with the needed parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        cfg, vocabulary = spec
        return GraphNNDecoder(model, cfg, vocabulary)

    def param_collection(self):
        """Return a :code:`dynet.ParameterCollection` object with the parameters.

        It is one of the prerequisites for Dynet save/load method.
        """
        return self.pc

