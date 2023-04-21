from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import base64
from style_transfer import run_style_transfer, image_loader, ContentLoss, StyleLoss, get_style_model_and_losses, gram_matrix
import torch


app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

@app.route('/transfer', methods=['POST'])
def transfer():
    if 'style' not in request.files or 'content' not in request.files:
        return jsonify({'error': 'Missing style or content image'}), 400

    style_image = request.files['style']
    content_image = request.files['content']
    print(type(style_image))
    print(type(content_image))

    if not allowed_file(style_image.filename) or not allowed_file(content_image.filename):
        return jsonify({'error': 'Invalid file type, allowed types are: png, jpg, jpeg'}), 400

    print('Running Style Transfer')
    output_tensor = run_style_transfer(style_image, content_image)
    print('Converting to Image')
    output_image = tensor_to_image(output_tensor)
    print('Saving Image')
    buffered = BytesIO()
    output_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({'image': img_str}), 200


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def tensor_to_image(tensor):
    #Helper function to convert a PyTorch tensor to a PIL image.
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = tensor.mul(255).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    return Image.fromarray(tensor)


if __name__ == '__main__':
    app.run()

