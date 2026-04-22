import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import io

def preprocess_image(image_bytes: bytes):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return tf(img).unsqueeze(0)

def generate_heatmap(model, image_tensor, target_layer):
    model.eval()

    activations = []
    gradients   = []

    # ✅ Capture both forward activations AND backward gradients via hooks
    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    pred_idx = output.argmax(dim=1).item()

    # Backward pass for the predicted class
    model.zero_grad()
    output[0, pred_idx].backward()

    # Remove hooks
    fwd_handle.remove()
    bwd_handle.remove()

    # ✅ Pool gradients over spatial dimensions
    pooled_grads = torch.mean(gradients[0], dim=[0, 2, 3])  # shape: (C,)
    act = activations[0].squeeze(0)                          # shape: (C, H, W)

    # ✅ Weight each activation map by its gradient
    for i in range(act.shape[0]):
        act[i, :, :] *= pooled_grads[i]

    # Average across channels → heatmap
    heatmap = torch.mean(act, dim=0).numpy()
    heatmap  = np.maximum(heatmap, 0)

    # Normalize to [0, 1]
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    return heatmap, pred_idx