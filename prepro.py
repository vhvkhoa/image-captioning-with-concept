from collections import Counter
import torch
from core.utils import save_json, load_json
from core.feature_extractor import FeatureExtractor
from core.dataset import CocoImageDataset

import numpy as np
import os
from tqdm import tqdm
import argparse

ANN_DIR = 'data/annotations'
ANN_FILES = {
    'train': 'captions_train2017.json',
    'val': 'captions_val2017.json',
    'test': ''
}
CONCEPT_FILES = {
    'train': 'train_concepts.json',
    'val': 'val_concepts.json',
    'test': 'test_concepts.json'
}

"""Parameters for pre-processing"""
parser = argparse.ArgumentParser(description='Pre-processing dataset.')

parser.add_argument('-p', '--phases', type=str, default='train,val,test', help='Phases in which you want to pre-process the dataset. '+
                                                'Phases should be seperated by commas. '+
                                                'Images of phase named <phase> should be placed in image/<phase>. '+
                                                'By default, we pre-process all splits of COCO-dataset.(e.g \'train,val,test\')')

parser.add_argument('-a', '--ann_files', type=str, default='')

parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size to be used for extracting features from images.')

parser.add_argument('-m', '--max_length', type=int, default=30, help='Max length, only be used to pre-process captions in training split of dataset. \n'+
                                                                     'Captions have more words than max_length will be removed.')

parser.add_argument('-t', '--word_count_threshold', type=int, default=1, 
                                                help='Words occur less than word_count_threshold times in the dataset '+
                                                '(only apply for training set) will be removed from vocabulary '+
                                                'and replaced by <UNK> token in the captions.')

parser.add_argument('-v', '--vocab_size', type=int, default=0, 
                                                help='Size of vocabulary. Vocabulary is made of vocab_size most frequent words in the dataset. '+
                                                'Leave it to default value means not using it.')

parser.add_argument('-e' '--encoder_name', type=str, default='resnet101', help='CNN model name used to extract features of images.'+
                                                             'It should be vgg or resnet followed by a number indicating number of layers in the model (e.g vgg16, vgg19, resnet50, resnet101).')

parser.add_argument('-n', '--tag_names_file', type=str, default='data/9k.names')

def _process_caption_data(phase, ann_file=None, max_length=None):
    if phase in ['val', 'train']:
        caption_data = load_json(ann_file)

        if phase == 'val':
            caption_data['type'] = 'caption'

        # id_to_filename is a dictionary such as {image_id: filename]} 
        id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}

        # data is a list of dictionary which contains 'captions', 'file_name' and 'image_id' as key.
        for i, annotation in enumerate(caption_data['annotations']):
            image_id = annotation['image_id']
            caption_data['annotations'][i]['file_name'] = id_to_filename[image_id]

        if phase == 'train':
            del_idx = []
            for i, annotation in enumerate(caption_data['annotations']):
                caption = annotation['caption']
                caption = caption.replace('.','').replace(',','').replace("'",'').replace('"','')
                caption = caption.replace('&','and').replace('(','').replace(')','').replace('-',' ')
                caption = ' '.join(caption.split())  # replace multiple spaces

                caption_data['annotations'][i]['caption'] = caption.lower()
                if max_length != None and len(caption.split(' ')) > max_length:
                    del_idx.append(i)

            # delete captions if size is larger than max_length
            print("The number of captions before deletion: %d" %len(caption_data['annotations']))
            for idx in sorted(del_idx, reverse=True):
                del caption_data['annotations'][idx]
            print("The number of captions after deletion: %d" %len(caption_data['annotations']))

        save_json(caption_data, os.path.join('data', phase, os.path.basename(ann_file)))


def _process_concept_data(phase, word_to_idx, concept_file, max_keep=20):
    concept_data = load_json(concept_file)
    concept_dict = {}
    num_tags, num_concepts = [], []

    for file_name, concepts in enumerate(concept_data):
        concepts = sorted([[tag, int(raw_prob[:-1])] for tag, raw_prob in concepts], key=lambda x: x[1], reverse=True)
        num_tags.append(len(concepts))
        concepts = ' '.join([concept[0] for concept in concepts[:max_keep]]).split(' ')
        concepts = [word_to_idx[concept] for concept in concepts]
        concept_dict[file_name] = concepts
        num_concepts.append(len(concepts))
    
    save_json(num_tags, '%s_num_tags.json' % phase)
    save_json(num_concepts, '%s_num_concepts.json' % phase)

    save_json(concept_dict, os.path.join('data', phase, os.path.basename(concept_file)))


def _build_vocab(captions_data, tag_names_data, threshold=1, vocab_size=0):
    annotations = captions_data['annotations']
    counter = Counter()
    max_len = 0
    for annotation in annotations:
        caption = annotation['caption']
        words = caption.split(' ')  # caption contrains only lower-case words
        for w in words:
            counter[w] += 1

        if len(caption.split(' ')) > max_len:
            max_len = len(caption.split(' '))
    
    tag_names = tag_names_data.replace('\n', ' ').split(' ')
    for name in tag_names:
        counter[name] += 1
        
    if vocab_size > 0:
        top_n_counter = [w for w, n in counter.most_common(vocab_size)]
        vocab = [word for word in counter if counter[word] >= threshold and word in top_n_counter]
    else:
        vocab = [word for word in counter if counter[word] >= threshold]

    print('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))
    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<UNK>': 3}
    idx = len(word_to_idx)
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1

    print("Max length of caption: ", max_len)
    return word_to_idx


def _build_caption_vector(captions_data, word_to_idx):
    annotations = captions_data['annotations']

    for i, annotation in enumerate(annotations):
        caption = annotation['caption']
        words = caption.split(' ') # caption contains only lower-case words
        cap_vec = [word_to_idx['<START>']]
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
            else:
                cap_vec.append(word_to_idx['<UNK>'])
        cap_vec.append(word_to_idx['<END>'])
        captions_data['annotations'][i]['vector'] = cap_vec

    print("Finished building caption vectors")
    return captions_data


def main():
    args = parser.parse_args()
    # phases to be processed.
    phases = [phase.strip() for phase in args.phases.split(',')]

    # annotation files and concept files to be processed
    ann_files = [os.path.join(ANN_DIR, ANN_FILES[phase]) for phase in phases]
    concept_files = [os.path.join(ANN_DIR, CONCEPT_FILES[phase]) for phase in phases]

    # batch size for extracting feature vectors.
    batch_size = args.batch_size

    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
    max_length = args.max_length

    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = args.word_count_threshold
    vocab_size = args.vocab_size

    # names list
    with open(args.tag_names_file, 'r') as f:
        tag_names_data = f.read()

    for phase, ann_file, concept_file in zip(phases, ann_files, concept_files):
        _process_caption_data(phase, ann_file=ann_file, max_length=max_length)

        if phase == 'train':
            captions_data = load_json('./data/train/captions_train2017.json')

            word_to_idx = _build_vocab(captions_data, tag_names_data, threshold=word_count_threshold, vocab_size=vocab_size)
            save_json(word_to_idx, './data/word_to_idx.json')

            new_captions_data = _build_caption_vector(captions_data, word_to_idx=word_to_idx)
            save_json(new_captions_data, ann_file)
        
        word_to_idx = load_json('./data/word_to_idx.json')
        _process_concept_data(phase, word_to_idx, concept_file)

    print('Finished processing caption data')

    feature_extractor = FeatureExtractor(model_name='resnet101', layer=3)
    for phase in phases:
        if not os.path.isdir('./data/%s/feats/' % phase):
            os.makedirs('./data/%s/feats/' % phase)

        image_paths = os.listdir('./image/%s/' % phase)
        dataset = CocoImageDataset(root='./image/%s/' % phase, image_paths=image_paths)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)

        for batch_paths, batch_images in tqdm(data_loader):
            feats = feature_extractor(batch_images).data.cpu().numpy()
            feats = feats.reshape(-1, feats.shape[1]*feats.shape[2], feats.shape[-1])
            for j in range(len(feats)):
                np.save('./data/%s/feats/%s.npy' % (phase, batch_paths[j]), feats[j])


if __name__ == "__main__":
    main()
