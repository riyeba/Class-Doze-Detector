
from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer
import torch
from mmengine.logging.history_buffer import HistoryBuffer
from numpy.core.multiarray import _reconstruct, scalar
import numpy
import builtins
import os
from mmengine.config import Config

# Debug: Check versions and environment
print("PyTorch version:", torch.__version__)
print("MMEngine version:", os.popen('pip show mmengine').read().split('Version: ')[1].split('\n')[0] if 'mmengine' in os.popen('pip list').read() else "Not installed")
print("MMAction2 version:", os.popen('pip show mmaction2').read().split('Version: ')[1].split('\n')[0] if 'mmaction2' in os.popen('pip list').read() else "Not installed")
print("TORCH_FORCE_WEIGHTS_ONLY_LOAD:", os.getenv('TORCH_FORCE_WEIGHTS_ONLY_LOAD'))

# Debug: Check current safe globals
print("Current safe globals:", torch.serialization.get_safe_globals())

# Add all problematic globals to safe globals
torch.serialization.add_safe_globals([
    HistoryBuffer,
    _reconstruct,
    numpy.dtype,
    builtins.getattr,
    scalar,
    numpy.ndarray
])

# Debug: Confirm safe globals were added
print("Updated safe globals:", torch.serialization.get_safe_globals())

# Debug: Check all unsupported globals in checkpoint
try:
    with open(r"D:\Class-Doze-Detector\mmaction2\work_dirs\tsn_sleeping\best_acc_top1_epoch_1.pth", 'rb') as f:
        unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(f)
        print("Unsupported globals in checkpoint:", unsafe_globals)
except Exception as e:
    print("Failed to check unsupported globals:", e)

# Configuration file
config_file = r"D:\Class-Doze-Detector\mmaction2\configs\recognition\tsn\tsn_sleeping.py"

# Checkpoint file
checkpoint_file = r"D:\Class-Doze-Detector\mmaction2\work_dirs\tsn_sleeping\best_acc_top1_epoch_1.pth"

# Video and label files
video_file = 'child2.mp4'
label_file = 'label_map.txt'

# Verify file paths
print("Config file exists:", os.path.exists(config_file))
print("Checkpoint file exists:", os.path.exists(checkpoint_file))
print("Video file exists:", os.path.exists(video_file))
print("Label file exists:", os.path.exists(label_file))

# Try loading with safe globals
try:
    with torch.serialization.safe_globals([HistoryBuffer, _reconstruct, numpy.dtype, builtins.getattr, scalar, numpy.ndarray]):
        model = init_recognizer(config_file, checkpoint_file, device='cpu')  # or device='cuda:0' if GPU is available
except Exception as e:
    print(f"Failed to load with safe globals: {e}")
    # Fallback: Load checkpoint manually with weights_only=False
    print("Falling back to weights_only=False (use only if checkpoint is trusted)")
    cfg = Config.fromfile(config_file)
    # Initialize model without checkpoint
    model = init_recognizer(config_file, None, device='cpu')  # Initialize without loading checkpoint
    # Load checkpoint manually
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)

# Run inference
pred_result = inference_recognizer(model, video_file)

# Process prediction results
pred_scores = pred_result.pred_score.tolist()
score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
top5_label = score_sorted[:5]

# Load labels
labels = open(label_file).readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in top5_label]

# Print results
print('The top-2 labels with corresponding scores are:')
for result in results[:2]:
    print(f'{result[0]}: {result[1]}')