import pytest
import torch
from numpy.testing import assert_allclose

from onnxruntime.capi.training.optim import lr_scheduler
from onnxruntime.capi.training.amp import loss_scaler
from onnxruntime.capi.training import pytorch_trainer_options as pt_options
from onnxruntime.capi.training import pytorch_trainer
from onnxruntime.capi.training import optim


@pytest.mark.parametrize("test_input", [
    ({}),
    ({'batch': {},
      'cuda': {},
      'distributed': {},
      'mixed_precision': {},
      'utils': {},
      '_internal_use': {}})
])
def testDefaultValues(test_input):
    ''' Test different ways of using default values for incomplete input'''

    expected_values = {
        'batch': {
            'gradient_accumulation_steps': 0
        },
        'cuda': {
            'device': None,
            'mem_limit': 0
        },
        'distributed': {
            'world_rank': 0,
            'world_size': 1,
            'local_rank': 0,
            'allreduce_post_accumulation': False,
            'enable_partition_optimizer': False,
            'enable_adasum': False
        },
        'lr_scheduler': None,
        'mixed_precision': {
            'enabled': False,
            'loss_scaler': None
        },
        'utils': {
            'grad_norm_clip': False
        },
        '_internal_use': {
            'frozen_weights': [],
            'enable_internal_postprocess': True,
            'extra_postprocess': None
        }
    }

    actual_values = pt_options.PytorchTrainerOptions(test_input)
    assert actual_values._validated_opts == expected_values


def testInvalidMixedPrecisionEnabledSchema():
    '''Test an invalid input based on schema validation error message'''

    expected_msg = 'must be of boolean type'
    actual_values = pt_options.PytorchTrainerOptions(
        {'mixed_precision': {'enabled': 1}})
    assert actual_values.mixed_precision[0].enabled[0] == expected_msg

def testTrainStepInfo():
    '''Test valid initializations of TrainStepInfo'''

    step_info = pytorch_trainer.TrainStepInfo(all_finite=True, epoch=1, step=2)
    assert step_info.all_finite is True
    assert step_info.epoch == 1
    assert step_info.step == 2

    step_info = pytorch_trainer.TrainStepInfo()
    assert step_info.all_finite is None
    assert step_info.epoch is None
    assert step_info.step is None


@pytest.mark.parametrize("test_input", [
    (-1),
    ('Hello'),
])
def testTrainStepInfoInvalidAllFinite(test_input):
    '''Test invalid initialization of TrainStepInfo'''
    with pytest.raises(AssertionError):
        pytorch_trainer.TrainStepInfo(all_finite=test_input)

    with pytest.raises(AssertionError):
        pytorch_trainer.TrainStepInfo(epoch=test_input)

    with pytest.raises(AssertionError):
        pytorch_trainer.TrainStepInfo(step=test_input)


@pytest.mark.parametrize("optim_name", [
    ('AdamOptimizer'),
    ('LambOptimizer'),
    ('SGDOptimizer')
])
def testOptimizerConfigs(optim_name):
    '''Test initialization of _OptimizerConfig and its extensions'''
    hyper_parameters={'lr':0.001}
    cfg = optim.config._OptimizerConfig(name=optim_name, hyper_parameters=hyper_parameters, param_groups=[])
    assert cfg.name == optim_name
    rtol = 1e-03
    assert_allclose(hyper_parameters['lr'], cfg.lr, rtol=rtol, err_msg="loss mismatch")
