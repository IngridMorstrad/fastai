#Code directly taken from NVIDIA apex: https://github.com/NVIDIA/apex
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def convert_network(network, dtype):
    """
    Converts a network's parameters and buffers to dtype.
    """
    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
            continue
        for param in module.parameters(recurse=False):
            if param is not None:
                if param.data.dtype.is_floating_point:
                    param.data = param.data.to(dtype=dtype)
                if param._grad is not None and param._grad.data.dtype.is_floating_point:
                    param._grad.data = param._grad.data.to(dtype=dtype)
        for buf in module.buffers(recurse=False):
            if buf is not None and buf.data.dtype.is_floating_point:
                buf.data = buf.data.to(dtype=dtype)
        if isinstance(module, (torch.nn.RNNBase, torch.nn.modules.rnn.RNNBase)):
            module.flatten_parameters()
    return network


def model_grads_to_master_grads(model_params, master_params, flat_master=False):
    """
    Copy model gradients to master gradients.

    Args:
        model_params:  List of model parameters (tensors with requires_grad=True).
        master_params:  List of FP32 master parameters cloned from model_params.
            If ``master_params`` was created with ``flat_master=True``,
            ``flat_master=True`` should also be supplied to this function.
    """
    if flat_master:
        master_params[0].grad.data.copy_(
            _flatten_dense_tensors([p.grad.data for p in model_params]))
    else:
        for model, master in zip(model_params, master_params):
            if model.grad is not None:
                if master.grad is None:
                    master.grad = torch.zeros_like(master.data)
                master.grad.data.copy_(model.grad.data)
            else:
                master.grad = None


def master_params_to_model_params(model_params, master_params, flat_master=False):
    """
    Copy master parameters to model parameters.

    Args:
        model_params:  List of model parameters (tensors with requires_grad=True).
        master_params:  List of FP32 master parameters cloned from model_params.
            If ``master_params`` was created with ``flat_master=True``,
            ``flat_master=True`` should also be supplied to this function.
    """
    if flat_master:
        for model, master in zip(model_params,
                                 _unflatten_dense_tensors(master_params[0].data, model_params)):
            model.data.copy_(master)
    else:
        for model, master in zip(model_params, master_params):
            model.data.copy_(master.data)
