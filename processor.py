import io
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    # 1. Convert to grayscale numpy array
    gray = np.array(pil_image.convert('L'))
    
    # 2. Apply CLAHE to equalize contrast and suppress heavy bone highlights
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # 3. Convert back to 3-channel RGB PIL Image
    rgb_img = Image.fromarray(equalized).convert('RGB')
    
    # 4. Standard ResNet transform with CenterCrop
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return tf(rgb_img).unsqueeze(0)
def generate_gradcam(model, image_tensor: torch.Tensor, original_pil_img: Image.Image) -> bytes:
    model.eval()

    # Ensure gradients can flow for Grad-CAM backpropagation
    for param in model.parameters():
        param.requires_grad = True

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Target the final convolutional layer of ResNet50
    target_layer = model.layer4[-1]
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)
    pred_idx = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    output[0, pred_idx].backward()

    # Clean up handles immediately
    fwd_handle.remove()
    bwd_handle.remove()

    # Extract tensors and release from computation graph
    pooled_grads = torch.mean(gradients[0].detach(), dim=[0, 2, 3])
    act = activations[0].detach().squeeze(0)

    # Calculate weighted feature map combination
    heatmap = torch.sum(pooled_grads.view(-1, 1, 1) * act, dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    # Match raw X-ray spatial dimensions
    orig_np = np.array(original_pil_img)
    h, w = orig_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to 8-bit array and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert BGR (OpenCV) to RGB (PIL standard)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Force original image to RGB array for blending
    if len(orig_np.shape) == 2 or orig_np.shape[2] == 1:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
    elif orig_np.shape[2] == 4:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGBA2RGB)

    # Overlay activation heatmap over raw X-ray scan
    overlay = cv2.addWeighted(orig_np, 0.6, heatmap_colored_rgb, 0.4, 0)

    # Return composite as PNG bytes
    res_img = Image.fromarray(overlay)
    img_byte_arr = io.BytesIO()
    res_img.save(img_byte_arr, format='PNG')

    return img_byte_arr.getvalue()
