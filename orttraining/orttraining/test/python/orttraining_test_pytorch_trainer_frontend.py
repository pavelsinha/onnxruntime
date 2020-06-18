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
      'device': {},
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
        'device': {
            'id': None,
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
    '''Test initialization of _OptimizerConfig'''
    hyper_parameters = {'lr': 0.001, 'alpha': 0.9}
    param_groups = [{'params': ['fc1.weight', 'fc2.weight'], 'alpha':.0}]
    cfg = optim.config._OptimizerConfig(
        name=optim_name, hyper_parameters=hyper_parameters, param_groups=param_groups)

    assert cfg.name == optim_name
    rtol = 1e-03
    assert_allclose(hyper_parameters['lr'],
                    cfg.lr, rtol=rtol, err_msg="lr mismatch")


@pytest.mark.parametrize("optim_name,hyper_parameters,param_groups", [
    ('AdamOptimizer', {'lr': -1}, []),  # invalid lr
    ('FooOptimizer', {'lr': 0.001}, []),  # invalid name
    ('SGDOptimizer', [], []),  # invalid type(hyper_parameters)
    (optim.config.Adam, {'lr': 0.003}, []),  # invalid type(name)
    ('AdamOptimizer', {'lr': None}, []),  # missing 'lr' hyper parameter
    ('SGDOptimizer', {'lr': 0.004}, {}),  # invalid type(param_groups)
    ('AdamOptimizer', {'lr': 0.005, 'alpha': 2}, [[]]), # invalid type(param_groups[i])
    ('AdamOptimizer', {'lr': 0.005, 'alpha': 2}, [{'alpha': 1}]), # missing 'params' at 'param_groups'
    ('AdamOptimizer', {'lr': 0.005}, [{'params': 'param1', 'alpha': 1}]), # missing 'alpha' at 'hyper_parameters'
])
def testOptimizerConfigsInvalidInputs(optim_name, hyper_parameters, param_groups):
    '''Test invalid initialization of _OptimizerConfig'''

    with pytest.raises(AssertionError):
        optim.config._OptimizerConfig(
            name=optim_name, hyper_parameters=hyper_parameters, param_groups=param_groups)


def testSGD():
    '''Test initialization of SGD'''
    cfg = optim.config.SGD()
    assert cfg.name == 'SGDOptimizer'

    rtol = 1e-05
    assert_allclose(0.001, cfg.lr, rtol=rtol, err_msg="lr mismatch")

    cfg = optim.config.SGD(lr=0.002)
    assert_allclose(0.002, cfg.lr, rtol=rtol, err_msg="lr mismatch")


def testAdam():
    '''Test initialization of Adam'''
    cfg = optim.config.Adam()
    assert cfg.name == 'AdamOptimizer'

    rtol = 1e-05
    assert_allclose(0.001, cfg.lr, rtol=rtol, err_msg="lr mismatch")
    assert_allclose(0.9, cfg.alpha, rtol=rtol, err_msg="alpha mismatch")
    assert_allclose(0.999, cfg.beta, rtol=rtol, err_msg="beta mismatch")
    assert_allclose(0.0, cfg.lambda_coef, rtol=rtol, err_msg="lambda_coef mismatch")
    assert_allclose(1e-8, cfg.epsilon, rtol=rtol, err_msg="epsilon mismatch")
    assert cfg.do_bias_correction == True, "lambda_coef mismatch"
    assert cfg.weight_decay_mode == True, "weight_decay_mode mismatch"


def testLamb():
    '''Test initialization of Lamb'''
    cfg = optim.config.Lamb()
    assert cfg.name == 'LambOptimizer'
    rtol = 1e-05
    assert_allclose(0.001, cfg.lr, rtol=rtol, err_msg="lr mismatch")
    assert_allclose(0.9, cfg.alpha, rtol=rtol, err_msg="alpha mismatch")
    assert_allclose(0.999, cfg.beta, rtol=rtol, err_msg="beta mismatch")
    assert_allclose(0.0, cfg.lambda_coef, rtol=rtol, err_msg="lambda_coef mismatch")
    assert cfg.ratio_min == float('-inf'), "ratio_min mismatch"
    assert cfg.ratio_max == float('inf'), "ratio_max mismatch"
    assert_allclose(1e-6, cfg.epsilon, rtol=rtol, err_msg="epsilon mismatch")
    assert cfg.do_bias_correction == True, "lambda_coef mismatch"
