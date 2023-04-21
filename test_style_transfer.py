import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from style_transfer import run_style_transfer, image_loader, ContentLoss, StyleLoss, get_style_model_and_losses, gram_matrix

from PIL import Image

import copy

# Unit tests

def test_image_loader():
    img_path = "test_image.jpg"
    img = Image.new("RGB", (512, 512), (255, 255, 255))
    img.save(img_path)
    tensor = image_loader(img_path)
    assert tensor.shape == (1, 3, 512, 512)

def test_ContentLoss():
    target = torch.randn(1, 3, 512, 512)
    loss_fn = ContentLoss(target)
    input = torch.randn(1, 3, 512, 512)
    output = loss_fn(input)
    assert output.shape == (1, 3, 512, 512)
    assert hasattr(loss_fn, 'loss')

def test_gram_matrix():
    temp = torch.randn(1, 3, 512, 512)
    gram = gram_matrix(temp)
    assert gram.shape == torch.Size([3, 3])
    
    temp = torch.randn(3, 3, 512, 512)
    gram = gram_matrix(temp)
    assert gram.shape == torch.Size([9, 9])

    
def test_StyleLoss():
    target_feature = torch.randn(1, 3, 512, 512)
    loss_fn = StyleLoss(target_feature)
    input = torch.randn(1, 3, 512, 512)
    output = loss_fn(input)
    assert output.shape == (1, 3, 512, 512)
    assert hasattr(loss_fn, 'loss')
    
def test_get_style_model_and_losses():
    cnn = models.vgg19(pretrained=True).features
    style_img = torch.randn(1, 3, 512, 512)
    content_img = torch.randn(1, 3, 512, 512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, device)
    assert len(style_losses) == 5
    assert len(content_losses) == 1
    assert isinstance(model, nn.Sequential)

def test_run_style_transfer():
    style_img = "style_image.jpg"
    content_img = "content_image.jpg"
    img1 = Image.new("RGB", (512, 512), (255, 255, 255))
    img2 = Image.new("RGB", (512, 512), (255, 255, 255))
    img1.save(style_img)
    img2.save(content_img)
    output = run_style_transfer(style_img, content_img)
    assert output.shape == (1, 3, 512, 512)

# Integration test

def test_style_transfer():
    style_img = "style_image.jpg"
    content_img = "content_image.jpg"
    img1 = Image.new("RGB", (512, 512), (255, 255, 255))
    img2 = Image.new("RGB", (512, 512), (255, 255, 255))
    img1.save(style_img)
    img2.save(content_img)
    output = run_style_transfer(style_img, content_img)
    assert output.shape == (1, 3, 512, 512)

if __name__ == '__main__':
    test_image_loader()
    test_ContentLoss()
    test_gram_matrix()
    test_StyleLoss()
    test_get_style_model_and_losses()
    test_run_style_transfer()
    test_style_transfer()
