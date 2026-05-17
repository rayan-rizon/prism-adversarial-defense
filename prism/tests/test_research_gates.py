"""
Research-standard regression tests for the CIFAR-native PRISM pipeline.

These tests avoid remote-scale artifacts and ImageNet downloads. They check
the invariants that must hold before a Vast.ai full run is worth launching.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import BACKBONE_INPUT_SIZE, BACKBONE_NUM_CLASSES, DATASET, MAX_DIM, N_SUBSAMPLE
from src.models.cifar_resnet import cifar_resnet18
from src.tamm.persistence_stats import (
    LOGIT_PROFILE_FEATURE_COUNT,
    compute_logit_profile_features,
)
from src.tamm.tda import TopologicalProfiler
from src.tamsh.experts import ExpertSubNetwork


class TestBackboneAndExpertShapes:
    def test_cifar_resnet_shape_matches_active_config(self):
        model = cifar_resnet18(num_classes=BACKBONE_NUM_CLASSES).eval()
        x = torch.randn(2, 3, BACKBONE_INPUT_SIZE, BACKBONE_INPUT_SIZE)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, BACKBONE_NUM_CLASSES)
        assert BACKBONE_NUM_CLASSES in {10, 100}
        assert DATASET in {'cifar10', 'cifar100'}

    def test_tamsh_expert_output_uses_dataset_classes(self):
        expert = ExpertSubNetwork(input_dim=32, output_dim=BACKBONE_NUM_CLASSES, hidden_dim=16)
        logits = expert(torch.randn(4, 32))
        assert logits.shape == (4, BACKBONE_NUM_CLASSES)


class TestTDADeterminism:
    def test_same_input_same_diagram(self):
        profiler = TopologicalProfiler(n_subsample=N_SUBSAMPLE, max_dim=MAX_DIM)
        rng = np.random.RandomState(123)
        act = rng.randn(256, 64).astype(np.float32)

        d1 = profiler.compute_diagram(act)
        d2 = profiler.compute_diagram(act)

        assert len(d1) == len(d2)
        for a, b in zip(d1, d2):
            a = np.asarray(a)
            b = np.asarray(b)
            assert a.shape == b.shape
            if a.size > 0:
                np.testing.assert_allclose(
                    np.sort(a.flatten()),
                    np.sort(b.flatten()),
                    rtol=0,
                    atol=1e-8,
                )


class TestFeatureContracts:
    def test_logit_profile_feature_block_is_finite_and_fixed_width(self):
        logits = np.array([4.0, 1.0, -2.0, 0.5], dtype=np.float32)

        features = compute_logit_profile_features(logits)

        assert features.shape == (LOGIT_PROFILE_FEATURE_COUNT,)
        assert np.all(np.isfinite(features))
        assert features.dtype == np.float32


class TestScriptPreflight:
    @pytest.mark.parametrize('script', ['run_vastai_full.sh', 'run_vastai_cifar100.sh'])
    def test_vastai_scripts_are_bash_syntax_valid(self, script):
        bash = shutil.which('bash')
        if bash is None:
            pytest.skip('bash is required for shell-script syntax preflight')

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        result = subprocess.run(
            [bash, '-n', os.path.join(root, script)],
            cwd=root,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr


class TestTableBuilderFixtures:
    def test_table_builder_reads_multiseed_and_tagged_outputs(self, tmp_path):
        exp = tmp_path / 'experiments'
        eval_dir = exp / 'evaluation'
        abl_dir = exp / 'ablation'
        eval_dir.mkdir(parents=True)
        abl_dir.mkdir(parents=True)

        fast = {
            'aggregate': {
                'FGSM': {'pool_TP': 9, 'pool_FN': 1, 'pool_FP': 1, 'pool_TN': 9},
                'PGD': {'pool_TP': 8, 'pool_FN': 2, 'pool_FP': 1, 'pool_TN': 9},
            },
            'metadata': {'dataset': 'cifar100'},
        }
        (eval_dir / 'results_cifar100_fast_n10_ms5.json').write_text(json.dumps(fast))

        ablation = {
            'aggregate': {
                'Full PRISM': {'mean_TPR': 0.90},
                'Ensemble-no-TDA': {'mean_TPR': 0.84},
            },
            'statistical_tests': {
                'Ensemble-no-TDA': {
                    'FGSM': {'mean_delta': 0.05, 'p_value': 0.04},
                    'PGD': {'mean_delta': 0.07, 'p_value': 0.03},
                }
            },
            'metadata': {'dataset': 'cifar100'},
        }
        (abl_dir / 'results_cifar100_ablation_multiseed.json').write_text(json.dumps(ablation))

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        from build_paper_tables import table_ablation, table_main_attacks

        main_tex = table_main_attacks(str(exp))
        ablation_tex = table_ablation(str(exp))

        assert 'CIFAR-100 & FGSM' in main_tex
        assert 'CIFAR-100 & Ensemble-no-TDA' in ablation_tex
        assert '+0.0600' in ablation_tex
