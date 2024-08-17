import numpy as np
import torch
from PIL import Image
from matplotlib import cm
from io import BytesIO


def get_gradcam_heatmap(model, img, target_layer, class_idx=None):
    model.eval()
    conv_output = None
    grad_output = None

    def forward_hook(module, input, output):
        nonlocal conv_output
        conv_output = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal grad_output
        grad_output = grad_out[0]

    hook = target_layer.register_forward_hook(forward_hook)
    backward_hook = target_layer.register_backward_hook(backward_hook)

    output = model(img)
    if class_idx is None:
        class_idx = torch.argmax(output, dim=1)

    model.zero_grad()
    class_output = output[:, class_idx]
    class_output.backward()

    hook.remove()
    backward_hook.remove()

    pooled_gradients = torch.mean(grad_output, dim=[0, 2, 3])
    for i in range(pooled_gradients.size(0)):
        conv_output[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(conv_output, dim=1).squeeze()
    heatmap = torch.clamp(heatmap, min=0)
    heatmap = heatmap / torch.max(heatmap)

    return heatmap.detach().cpu().numpy(), class_idx.item()

def generate_gradcam_image(img_shape, heatmap, predicted_class_idx, class_names, alpha=0.4):
    img = np.zeros((*img_shape, 3))

    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((img_shape), Image.Resampling.LANCZOS)
    heatmap = np.array(heatmap)

    heatmap_colored = cm.jet(heatmap)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

    superimposed_img = heatmap_colored * alpha + img
    superimposed_img = np.uint8(superimposed_img)

    result_img = Image.fromarray(superimposed_img)

    img_io = BytesIO()
    result_img.save(img_io, format='PNG')
    img_io.seek(0)

    return img_io, class_names[predicted_class_idx]