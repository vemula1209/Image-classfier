# Imports here
import torch
import sys
import json
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch.nn.functional as F
from contextlib import contextmanager


def load_checkpoint(file_dir):  # Load pre-trained model checkpoint
    '''
        A function that loads a checkpoint and rebuilds the model
    '''
    # Load saved pre-trained model checkpoint
    checkpoint = torch.load(file_dir)

    # Get model architecture
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Model arch not found.")

    # Feature parameters
    for x in model.parameters():
        x.requires_grad = False

    # Load model parameters
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image

    # Process a PIL image for use in a PyTorch model
    img = Image.open(image)

    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    img_processed = transformations(img)

    return img_processed


def predict(img, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    # Turn on evaluation for model
    model.eval()

    # Turn on Cuda if available
    if gpu:
        model.cuda()
        img = img.cuda()

    # Process image into inputs and run through model
    outputs = model.forward(Variable(img.unsqueeze(0), volatile=True))

    # Get Probabilities and Classes
    probs, classes = outputs.topk(topk)
    probs = probs.exp().data.numpy()[0]
    classes = classes.data.numpy()[0]
    class_keys = {x: y for y, x in model.class_to_idx.items()}
    classes = [class_keys[i] for i in classes]

    return probs, classes


if __name__ == '__main__':
    '''
    This script predicts a label based on the model and image given
    as an argument when running the script.
    '''
    # Get arguments
    img_dir = sys.argv[1]
    check_point_path = sys.argv[2]
    args = sys.argv[:]

    # Number of predictions 'top_k' argument, else 5
    if "--top_k" in args:
        top_k = args[args.index("--top_k") + 1]
    else:
        top_k = 5

    # JSON file mapping image folder index numbers to labels
    if "--category_names" in args:
        category_names = args[args.index("--category_names") + 1]
    else:
        category_names = "cat_to_name.json"

    with open(category_names, 'r') as f:
        category_dict = json.load(f)

        # If '--gpu' is passed, enable Cuda features for PyTorch
    if "--gpu" in args and torch.cuda.is_available():
        gpu = True
        print('\nRunning GPU...')
    elif "--gpu" in args and not torch.cuda.is_available():
        gpu = False
        print('\nError: Cuda not available but --gpu was set.')
        print('Running CPU...\n')
    else:
        gpu = False
        print('\nRunning CPU...\n')

    # Load model
    model = load_checkpoint(check_point_path)

    # Process image and predict label via model
    img = process_image(img_dir)

    # Run image through model
    probs, classes = predict(img, model, gpu)

    # Display probabilities and Classes

    y = [category_dict.get(i) for i in classes[::]]
    x = np.array(probs)

    print("\n\n**Results from image {} using pretrained model checkpoint {}**".format(img_dir, check_point_path))
    print("\nProbabilies: {}".format(x))
    print("Classes: {}\n\n".format(y))
    print("End of program...")