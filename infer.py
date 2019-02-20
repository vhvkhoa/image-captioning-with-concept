import cPickle as pickle
from tensorflow import flags
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
from core.utils import evaluate

import argparse
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.dataset import CocoCaptionDataset

parser = argparse.ArgumentParser(description='Train model.')

"""Model's parameters"""
parser.add_argument('test_checkpoint', type=str, 'Path to a checkpoint used to infer.') 
parser.add_argument('word_to_idx_dict', type=str, 'Path to pickle file contained dictionary of words and their corresponding indices.')

parser.add_argument('--image_feature_size', type=int, default=196, 'Multiplication of width and height of image feature\'s dimension, e.g 14x14=196 in the original paper.')
parser.add_argument('--image_feature_depth', type=int, default=1024, 'Depth dimension of image feature, e.g 512 if you extract features at conv-5 of VGG-16 model.')
parser.add_argument('--lstm_hidden_size', type=int, default=1536, 'Hidden layer size for LSTM cell.')
parser.add_argument('--time_steps', type=int, default=31, 'Number of time steps to be iterating through.')
parser.add_argument('--embed_dim', type=int, default=512, 'Embedding space size for embedding tokens.')
parser.add_argument('--beam_size', type=int, default=3, 'Beam size for inference phase.')
parser.add_argument('--dropout', type=float, default=0.5, 'Dropout portion.')
parser.add_argument('--prev2out', action='store_true', default=True, 'Link previous hidden state to output.')
parser.add_argument('--ctx2out', action='store_true', default=True, 'Link context features to output.')
parser.add_argument('--enable_selector', action='store_true', default=True, 'Enable selector to determine how much important the image context is at every time step.')

"""Other parameters"""
parser.add_argument('--device', type=str, default='cuda:0', help='Device to be used for training model.')
parser.add_argument('--att_vis', action='store_true', default=False, 'Attention visualization, will show attention masks of every word.') 
parser.add_argument('--image_id_file', type=str, default='./data/val/captions_val2017.json')
parser.add_argument('--concept_file', type=str, default='./data/val/val_concepts.json')
parser.add_argument('--batch_size', type=int, default=128, 'Number of examples per mini-batch.')

def main():
    args = parser.parse_args()
    # load dataset and vocab
    test_data = CocoCaptionDataset(args.image_id_file,
                                  concept_file=args.concept_file, split='test')
    word_to_idx = train_data.get_vocab_dict()
    # load val dataset to print out scores every epoch

    model = CaptionGenerator(feature_dim=[args.image_feature_size, args.image_feature_depth], 
                                    num_tags=23, embed_dim=args.embed_dim,
                                    hidden_dim=args.lstm_hidden_size, prev2out=args.prev2out, len_vocab=len(word_to_idx),
                                    ctx2out=args.ctx2out, enable_selector=args.enable_selector, dropout=args.dropout).to(device=args.device)

    solver = CaptioningSolver(model, word_to_idx, n_time_steps=args.time_steps, batch_size=args.batch_size,
                                    beam_size=args.beam_size, optimizer=args.optimizer, 
                                    learning_rate=args.learning_rate, metric=args.metric,
                                    eval_every=args.eval_steps,
                                    checkpoint=args.checkpoint, checkpoint_dir=args.checkpoint_dir, 
                                    log_path=args.log_path, device=args.device)

    solver.test(data, test_dataset=test_data)


if __name__ == "__main__":
    main()
