import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image

import copy


def image_loader(image):
    """Helper function to load an image into a PyTorch tensor."""
    loader = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])
    # Load image and convert it to tensor
    image = Image.open(image)
    # Add batch dimension to tensor and return
    return loader(image).unsqueeze(0)


class ContentLoss(nn.Module):
    """Class representing the content loss function."""

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    """Helper function to calculate the Gram matrix of a tensor."""
    batch_size, channel, height, width = input.size()
    features = input.view(batch_size * channel, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * channel * height * width)


class StyleLoss(nn.Module):
    """Class representing the style loss function."""

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input


def get_style_model_and_losses(cnn, style_img, content_img, device):
    """Helper function to create a CNN model and initialize the loss
functions."""
    cnn = copy.deepcopy(cnn)

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []

    model = nn.Sequential()

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i],
StyleLoss):
            break

    model = model[:(i+1)]

    return model, style_losses, content_losses


def run_style_transfer(style_image, content_image, num_steps=300,
style_weight=1000000, content_weight=1):
    print('Get Device')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Pretrained model')
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    print('load Images')
    style_img = image_loader(style_image)
    content_img = image_loader(content_image)
    print('Get model and losses')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
style_img, content_img, device)
    print('Optimizer')
    optimizer = optim.LBFGS([content_img.requires_grad_()])

    print('Running style transfer...')
    run = [0]
    while run[0] <= num_steps:
        print('Step {}'.format(run[0]))

        def closure():
            content_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(content_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f'run {run}')
                print(f'Style Loss: {style_score.item()} Content Loss: {content_score.item()}')
                print()

            return style_score + content_score

        optimizer.step(closure)

    content_img.data.clamp_(0, 1)

    return content_img

