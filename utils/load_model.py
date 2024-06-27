import onnxruntime as ort


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_onnx_model(local_model_path: str):
    return ort.InferenceSession(local_model_path)
