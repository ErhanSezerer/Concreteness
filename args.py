import argparse

RANDOM_STATE = 42


# ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




parser = argparse.ArgumentParser(description='Concreteness Tests')
#BERT arguments
parser.add_argument('--MAX_LEN', type=int, default=512)
parser.add_argument('--num_label', type=int, default=2)
parser.add_argument('--bert_sep', type=str, default="<-SEP->")
parser.add_argument('--target_class', default=1, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--momentum', default=0.9)


#other arguments
parser.add_argument('--checkpoint_dir', default='./models')
parser.add_argument('--data_dir', default='/media/darg1/Data/Projects/Concreteness/data')
parser.add_argument('--image_path', default='/media/darg1/5EDD3D191C555EB5/wikimedia_dataset')
parser.add_argument('--use_gpu', type=str2bool, default=True)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--model_name', type=str, default="bert_chess_classifier_1")
parser.add_argument('--model_name_bert', type=str, default="bert_chess_classifier_1")
parser.add_argument('--model_name_resnet', type=str, default="bert_chess_classifier_1")
parser.add_argument('--classifier', type=str, default="bert") #options: bert, distilbert, resnet
parser.add_argument('--print_preds', type=str2bool)

parser.add_argument('--mode', default='concreteness', type=str, help="(options ['concreteness', 'wiki'] defaults to 'concreteness')")
parser.add_argument('--feature', default='caption', type=str, help="(options ['caption', 'description'] defaults to 'caption')")




args = parser.parse_args()
