import pytest
import torch

from vektr_trainer.features import EMPTY, fen_features
from vektr_trainer.model import Net, quantise
from vektr_trainer.train import targets
from vektr_trainer.vknn import check_fens, evaluate


def test_float_model_and_integer_evaluation_are_the_same_function():
    torch.manual_seed(0)
    net = Net()
    net.clip_()
    q = quantise(net)
    for fen in check_fens()[:64]:
        idx, stm = fen_features(fen)
        x = torch.tensor([idx + [EMPTY] * (32 - len(idx))])
        s = torch.tensor([stm])
        assert abs(net(x, s).item() * 400 - evaluate(q, fen)) <= 8


def test_quantised_values_are_inside_the_valid_ranges():
    net = Net()
    [p.data.mul_(100) for p in net.parameters()]
    net.clip_()
    q = quantise(net)
    assert abs(q.ft_w).max() <= 505 and abs(q.ft_b).max() <= 505 and abs(q.out_w).max() <= 127


def test_targets_blend_score_and_result_and_ignore_unknown_results():
    t = targets(torch.tensor([0.0, 400.0, 400.0]), torch.tensor([255, 2, 0]), 0.75)
    assert t.tolist() == pytest.approx([0.5, 0.75 * 0.7311 + 0.25, 0.75 * 0.7311], abs=1e-3)
