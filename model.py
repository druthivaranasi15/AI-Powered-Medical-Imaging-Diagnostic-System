import torch
import torchvision.models as models
import torch.nn as nn
import os

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
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    classes = checkpoint.get('classes', ['NORMAL', 'PNEUMONIA'])
    print(f"✅ Model loaded | Classes: {classes} | "
          f"Test acc: {checkpoint.get('test_accuracy', 'N/A')}")

    return model, classes