import onnxruntime as ort


def get_onnx_model(local_model_path: str):
    return ort.InferenceSession(local_model_path)
