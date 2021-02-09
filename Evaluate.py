import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
# from data_loader import get_loader
# import requests
# import pandas as pd
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import json

# path = '/home/khaaq/Documents/COCO_KarepathyData2014/annotations/'
# captionjson = pd.read_json('/home/khaaq/Documents/COCO_KarepathyData2014/annotations/captions_val2014.json', lines=True)
# print(captionjson)
# with open(path + 'captions_val2014.json', 'r') as json_file:
#      captionjson = json.load(json_file)
# pdDF= pd.DataFrame(captionjson)
# print(pdDF)
#     for p in captionjson['info']:
#         print('info', p)
#     age = ages.get(person, 0)
    # for p in data['people']:
    #     print('Name: ' + p['name'])
    #     print('Website: ' + p['website'])
    #     print('From: ' + p['from'])
    #     print('')
# image_path = '/home/khaaq/Documents/COCOTorch_Yunjey/ResizeTest2014/COCO_val2014_000000000486.jpg'

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters


    encoder.load_state_dict(torch.load(os.path.join('./Models', args.encoder_path)))
    decoder.load_state_dict(torch.load(os.path.join('./Models', args.decoder_path)))

    # Prepare an image
    for images in valimages:




    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    


    # Generate a caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(sentence)

    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    
    plt.show()


def cal_bleu_score(dataset, model, source_vocab, target_vocab):
    targets = []
    predictions = []
 
    for i in range(len(dataset)):
        target = vars(test_data.examples[i])['trg']
        predicted_words = predict(i, model, source_vocab, target_vocab, dataset)
        predictions.append(predicted_words[1:-1])
        targets.append([target])
 
    print(f'BLEU Score: {round(bleu_score(predictions, targets) * 100, 2)}')

source_vocab = args.vocab_path
target_vocab

cal_bleu_score(dataset, model, source_vocab, target_vocab)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valImage', type=str, default = '/home/khaaq/Documents/COCOTorch_Yunjey/ResizeTest2014/COCO_val2014_000000000536.jpg', help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='/home/khaaq/Documents/COCOTorch_Yunjey/Models/encoder-10-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='/home/khaaq/Documents/COCOTorch_Yunjey/Models/decoder-10-3000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='/home/khaaq/Documents/COCOTorch_Yunjey/vocab.pkl', help='path for vocabulary wrapper')
    # parser.add_argument('--caption_path', type=str, default='/home/khaaq/Documents/COCO_KarepathyData2014/annotations/captions_val2014.json', help='path for train annotation json file')
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)

