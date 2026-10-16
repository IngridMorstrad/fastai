"""GradCAM and Grad-CAM++ visualization for fastai vision models.

Provides class-discriminative heatmap generation to highlight regions
that contribute most to a model's prediction for a given class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['GradCAM', 'GradCAMPP', 'show_gradcam']


def _find_last_conv2d(model):
    "Find the last Conv2d layer in a model."
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


class GradCAM:
    "Computes GradCAM heatmaps for a given model and target convolutional layer."

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or _find_last_conv2d(model)
        self._hooks = []
        if self.target_layer is None:
            raise ValueError("No Conv2d layer found in model. Please specify target_layer.")
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        "Register forward and backward hooks on the target layer."
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def compute(self, input_tensor, class_idx=None):
        """Compute GradCAM heatmap for the given input.

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W) or (C, H, W).
            class_idx: Target class index. If None, uses the predicted class.

        Returns:
            Heatmap tensor of shape (H_input, W_input) with values in [0, 1].

        Raises:
            ValueError: If input_tensor has batch size > 1.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        if input_tensor.shape[0] != 1:
            raise ValueError(
                f"GradCAM only supports single-image inputs (batch size 1), "
                f"but got batch size {input_tensor.shape[0]}."
            )

        input_tensor = input_tensor.detach().requires_grad_(True)
        was_training = self.model.training
        self.model.eval()
        try:
            output = self.model(input_tensor)

            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            self.model.zero_grad()
            target = output[0, class_idx]
            target.backward()
        finally:
            if was_training:
                self.model.train()

        # Compute weights via global average pooling of gradients
        weights = self._compute_weights()

        # Weighted combination of feature maps
        cam = self._compute_cam(weights)

        # Upsample to input spatial size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = self._upsample(cam, h, w)

        return cam

    def _compute_weights(self):
        "Compute weights from gradients using global average pooling."
        # gradients shape: (1, C, H, W)
        return self.gradients.mean(dim=(2, 3), keepdim=True)

    def _compute_cam(self, weights):
        "Compute the weighted combination of activation maps."
        # activations shape: (1, C, H, W), weights shape: (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0)

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam

    def _upsample(self, cam, h, w):
        "Upsample the heatmap to the input spatial dimensions."
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
        return cam

    def remove(self):
        "Remove all hooks."
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()


class GradCAMPP(GradCAM):
    "Computes Grad-CAM++ heatmaps with improved weighting using higher-order gradients."

    def _compute_weights(self):
        "Compute Grad-CAM++ alpha weights using second and third order gradient information."
        grads = self.gradients  # (1, C, H, W)

        # Second and third powers of gradients
        grads_2 = grads.pow(2)
        grads_3 = grads.pow(3)

        # Sum of activations across spatial dimensions
        sum_activations = self.activations.sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Compute alpha (denominator with eps for numerical stability)
        eps = 1e-7
        denom = 2.0 * grads_2 + sum_activations * grads_3 + eps

        # Alpha coefficients
        alpha = grads_2 / denom  # (1, C, H, W)

        # Only consider pixels where gradient is positive
        positive_grads = F.relu(grads)

        # Weights: sum of (alpha * relu(gradient)) across spatial dims
        weights = (alpha * positive_grads).sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        return weights


def show_gradcam(model, img_tensor, class_idx=None, layer=None, method='gradcam', figsize=(8, 8)):
    """Show GradCAM/Grad-CAM++ heatmap overlaid on the input image.

    Args:
        model: A PyTorch model (nn.Module) or a fastai Learner (with .model attribute).
        img_tensor: Input image tensor of shape (C, H, W) or (1, C, H, W).
        class_idx: Target class index. If None, uses the predicted class.
        layer: Target convolutional layer. If None, uses the last Conv2d.
        method: 'gradcam' or 'gradcampp'.
        figsize: Matplotlib figure size.

    Returns:
        matplotlib Figure object.
    """
    # Validate method parameter
    valid_methods = ('gradcam', 'gradcampp')
    if method not in valid_methods:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of {valid_methods}."
        )

    # Support both raw models and Learner objects
    if hasattr(model, 'model'):
        net = model.model
    else:
        net = model

    cam_cls = GradCAMPP if method == 'gradcampp' else GradCAM
    cam_obj = cam_cls(net, target_layer=layer)

    try:
        heatmap = cam_obj.compute(img_tensor, class_idx=class_idx)
    finally:
        cam_obj.remove()

    # Prepare image for display
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    img_np = img_tensor.detach().cpu().numpy()

    # Convert CHW to HWP for display
    if img_np.shape[0] in (1, 3):
        img_np = np.transpose(img_np, (1, 2, 0))
    # Normalize image to [0, 1] for display
    img_min, img_max = img_np.min(), img_np.max()
    if img_max - img_min > 0:
        img_np = (img_np - img_min) / (img_max - img_min)

    heatmap_np = heatmap.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_np if img_np.shape[-1] == 3 else img_np.squeeze(), cmap='gray' if img_np.ndim == 2 or img_np.shape[-1] == 1 else None)
    ax.imshow(heatmap_np, cmap='jet', alpha=0.5)
    ax.set_title(f'{"Grad-CAM++" if method == "gradcampp" else "Grad-CAM"} (class {class_idx})')
    ax.axis('off')
    plt.tight_layout()

    return fig
