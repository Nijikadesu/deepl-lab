import torch
import numpy as np
from typing import Type


def validate_model(Net: Type[torch.nn.Module]):
    """freezing the random seed"""
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True

    """importing the model"""
    net = Net().cpu()
    dummy_input = torch.randn(1, 1, 28, 28)
    try:
        output = net(dummy_input).detach().cpu().numpy().tobytes()
    except Exception as e:
        print(f"Failed! Model check failed with error: {e}")
        print("Please make sure that the model is correctly implemented.")
        return
    ground_truth = (
        b'j\xa9L<\x18\xab\xd5\xbb\x04\xa5\xe0=B\xe9z=\x84\x0e\x8e;\xa6"K=X!\xf0\xbd\x035\x1b>R*\x08\xbe`\x80N\xbb'
    )

    if output == ground_truth:
        print("Success! Model check passed.")
    else:
        print("Failed! Model check failed.")
        print(f"Expected: {np.frombuffer(ground_truth, dtype=np.float32)}")
        print(f"Got: {np.frombuffer(output, dtype=np.float32)}")
