import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference
import torch


class Parsing:
    def __init__(self, gpu_id: int | str = "cpu"):
        self.device = self._resolve_device(gpu_id)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if self.device.type == "cuda":
            session_options.add_session_config_entry('gpu_id', str(self.device.index or 0))
        self.session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), 'ckpt/humanparsing/parsing_atr.onnx'),
                                            sess_options=session_options, providers=['CPUExecutionProvider'])
        self.lip_session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), 'ckpt/humanparsing/parsing_lip.onnx'),
                                                sess_options=session_options, providers=['CPUExecutionProvider'])

    @staticmethod
    def _resolve_device(gpu_id):
        if isinstance(gpu_id, torch.device):
            return gpu_id if gpu_id.type == "cpu" or torch.cuda.is_available() else torch.device("cpu")
        device_name = str(gpu_id).lower()
        if device_name == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        if device_name.startswith("cuda"):
            return torch.device(device_name)
        return torch.device(f"cuda:{gpu_id}")
        

    def __call__(self, input_image):
        # torch.cuda.set_device(self.gpu_id)
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask
