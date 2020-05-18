import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as transforms

class FeaturesHook():
    def __init__(self, module): 
        self.hook = module.register_forward_hook(self.hook_features)
    
    def hook_features(self, module, input, output): 
        self.features = output.cpu().data.numpy()

def getFCWeights(model):
    fc_layer = model._modules.get(list(model._modules.keys())[-1])
    
    fc_weights = list(fc_layer.parameters())[0]
    fc_weights = np.squeeze(fc_weights.cpu().data.numpy())
    return fc_weights

def visualizeCAM(model, image):
    image = image.resize((256, 256))
    model.eval()
    model.cuda()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image).cuda()

    conv = model._modules.get(list(model._modules.keys())[-3])
    hook = FeaturesHook(conv)

    prediction = model(image_tensor.unsqueeze(0))
    class_idx  = torch.sort(prediction, descending=True)[1][0][0].item()
    fc_weights = getFCWeights(model)[class_idx]

    features = hook.features
    c, h, w  = features.shape[1:]
    cam = fc_weights.dot(features.reshape((c, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, image.size)

    # visualize Class Activation Map
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.imshow(cam, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.show()