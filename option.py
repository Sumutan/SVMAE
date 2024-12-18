import argparse

parser = argparse.ArgumentParser(description='RTFM')
parser.add_argument('--exp-name', type=str, default=None, help='exp-name，also viz_name')
# dataset config
parser.add_argument('--use_dic_gt', action='store_true', default=False, help='get GrandTruth from a pickle file')
parser.add_argument('--abn_curve_save_root', type=str, default='./curve', help='folder for abn_curve_savepath')
parser.add_argument('--dataset', default='ucf', help='dataset to train on (shanghai, ucf, ped2, violence, TE2)')
# parser.add_argument('--alignment_method', type=str, default='add', choices=['add', 'cut'],
#                     help='the alignment method to dealwith superfluous frames')
parser.add_argument('--sampling', type=int, default=1, help='sampling of videos')

# feature config
parser.add_argument('--feature-group', default='both', choices=['both', 'vis', 'text'],
                    help='feature groups used for the model')
parser.add_argument('--fusion', type=str, default='concat',help='how to fuse vis and text features')
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d', 'videomae', 'clip'])
parser.add_argument('--feature-size', type=int, default=2048, help='size of vis feature (default: 2048)')
parser.add_argument('--feat-extractor-B', default='clip', choices=['i3d', 'c3d', 'videomae', 'clip'])
parser.add_argument('--feature-size-B', type=int, default=768, help='size of vis feature (default: 768(CLIP-L))')
parser.add_argument('--caption-extractor', default='swinBERT', choices=['swinBERT', 'clip'])
parser.add_argument('--emb_dim', type=int, default=768, help='dimension of text embeddings')
parser.add_argument('--rgb-list', default=None, help='list of rgb features ')
parser.add_argument('--test-rgb-list', default=None, help='list of test rgb features ')
parser.add_argument('--vis_feature_path_B_folder', default=None, help='folder path of rgb features(B) ')
parser.add_argument('--copy2SSD_path', default=None, help='moved dataset to the SSD to speed up reading')

# parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
parser.add_argument('--gt', default=None, help='file of ground truth ')
parser.add_argument('--gpus', default=1, type=int, choices=[0], help='gpus')
parser.add_argument('--lr', type=str, default='[0.001]*15000', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=32, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--model-name', default='rtfm', help='name to save model')
parser.add_argument('--pretrained-model', default=None, help='ckpt for pretrained model')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
parser.add_argument('--plot-best',  action='store_true', default=False, help='whether to plot anomaly curve')


parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 4869)')
parser.add_argument('--max-epoch', type=int, default=1000, help='maximum iteration to train (default: 1000)')
parser.add_argument('--normal_weight', type=float, default=1, help='weight for normal loss weights')
parser.add_argument('--abnormal_weight', type=float, default=1, help='weight for abnormal loss weights')
parser.add_argument('--aggregate_text', action='store_true', default=False, help='whether to aggregate text features')
parser.add_argument('--extra_loss', action='store_true', default=False, help='whether to use extra loss')
parser.add_argument('--save_test_results', action='store_true', default=False, help='whether to save test results')
parser.add_argument('--alpha', type=float, default=0.0001, help='weight for RTFM loss')
parser.add_argument('--emb_folder', type=str, default='sent_emb_n',
                    help='folder for text embeddings, used to differenciate different swinbert pretrained models')
