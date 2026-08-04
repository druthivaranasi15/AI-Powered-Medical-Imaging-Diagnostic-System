import os
import torch
import torch.nn as nn
import torchvision.models as models

def load_medical_model():
    model = models.resnet50(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )

    pth_path = os.path.join(os.path.dirname(__file__), 'pneumonia_resnet50.pth')
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model weight file not found at {pth_path}")

    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))

    classes = ['NORMAL', 'PNEUMONIA']
    state_dict = checkpoint

    # Extract state_dict if nested under common checkpoint keys
    if isinstance(checkpoint, dict):
        for key in ['model_state_dict', 'state_dict', 'model', 'net']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break

        if 'classes' in checkpoint:
            classes = checkpoint['classes']

    # Remove potential 'module.' prefix from DataParallel training
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    # Load state dict with strict set to False to bypass missing non-critical keys
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    print(f"✅ Model loaded successfully | Classes: {classes}")
    return model, classes
