from typing import Tuple
import fitlog
use_fitlog = False
if not use_fitlog:
    fitlog.debug()
fitlog.set_log_dir('logs')
load_dataset_seed = 100
fitlog.add_hyper(load_dataset_seed,'load_dataset_seed')
fitlog.set_rng_seed(load_dataset_seed)

import sys
sys.path.append('../')

import argparse
import collections

from paths import *
from utils import print_info
from utils import get_peking_time
from utils import norm_static_embedding
from load_data import *
from load_data import load_toy_ner
from V1.models import BERT_SeqLabel
from V1.add_lattice import equip_chinese_ner_with_lexicon, equip_chinese_tn_with_lexicon_and_rule
from V1.models import Lattice_Transformer_SeqLabel, Transformer_SeqLabel

from fastNLP import logger
from fastNLP import LRScheduler
from fastNLP import LossInForward, CrossEntropyLoss
from fastNLP import FitlogCallback
from fastNLP.core import Trainer
from fastNLP.core import Callback
from fastNLP_module import BertEmbedding
from fastNLP.core.metrics import SpanFPreRecMetric,AccuracyMetric
from fastNLP.core.callback import WarmupCallback,GradientClipCallback,EarlyStopCallback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#
#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))
# warnings.showwarning = warn_with_traceback


parser = argparse.ArgumentParser()
# performance inrelevant
parser.add_argument('--update_every',type=int,default=1)
parser.add_argument('--status',choices=['train','test'],default='train')
parser.add_argument('--use_bert', type=int, default=1)
parser.add_argument('--only_bert',type=int,default=0)
parser.add_argument('--use_rule', type=int, default=1)
parser.add_argument('--use_lexicon', type=int, default=1)
parser.add_argument('--fix_bert_epoch',type=int,default=5)
parser.add_argument('--after_bert', default='mlp', choices=['lstm', 'mlp'])
parser.add_argument('--msg', default='11266')
parser.add_argument('--train_clip', default=False, help='是不是要把train的char长度限制在200以内')
parser.add_argument('--device', default='0')
parser.add_argument('--debug', default=0, type=int)
parser.add_argument('--gpumm', default=False, help='查看显存')
parser.add_argument('--see_convergence', default=False)
parser.add_argument('--see_param', default=False)
parser.add_argument('--test_batch', default=-1)
parser.add_argument('--seed', default=1080956, type=int)
parser.add_argument('--test_train', default=False)
parser.add_argument('--number_normalized', type=int, default=0,
                    choices=[0,1,2,3], help='0不norm，1只norm char,2norm char和bigram，3norm char，bigram和lattice')
parser.add_argument('--lexicon_name', default='yj', choices=['lk','yj'])
parser.add_argument('--use_pytorch_dropout', type=int, default=0)

parser.add_argument('--char_min_freq',default=1,type=int)
parser.add_argument('--bigram_min_freq',default=1,type=int)
parser.add_argument('--lattice_min_freq',default=1,type=int)
parser.add_argument('--only_train_min_freq',default=True)
parser.add_argument('--only_lexicon_in_train',default=False)
parser.add_argument('--word_min_freq',default=1,type=int)

# hyper of training
parser.add_argument('--early_stop', default=5, type=int)
parser.add_argument('--epoch', default=20, type=int)
parser.add_argument('--batch', default=10, type=int)
parser.add_argument('--optim', default='sgd', help='sgd|adam')
parser.add_argument('--lr', default=6e-4, type=float)
parser.add_argument('--bert_lr_rate', default=0.05, type=float)
parser.add_argument('--embed_lr_rate', default=1, type=float)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--init', default='uniform', help='norm|uniform')
parser.add_argument('--self_supervised', default=False)
parser.add_argument('--weight_decay', default=0,type=float)
parser.add_argument('--norm_embed', default=True)
parser.add_argument('--norm_lattice_embed', default=True)

parser.add_argument('--warmup', default=0.1, type=float)

# hyper of model
# parser.add_argument('--use_bert',type=int,default=1)
parser.add_argument('--model', default='transformer', help='lstm|transformer')
parser.add_argument('--lattice', default=1, type=int)
parser.add_argument('--use_bigram', default=0, type=int)
parser.add_argument('--use_char_lstm', default=0, type=int)
parser.add_argument('--hidden', default=-1, type=int)
parser.add_argument('--ff', default=3,type=int)
parser.add_argument('--layer', default=1,type=int)
parser.add_argument('--head', default=8,type=int)
parser.add_argument('--head_dim',default=20,type=int)
parser.add_argument('--scaled',default=False)
parser.add_argument('--ff_activate',default='relu',help='leaky|relu')

parser.add_argument('--k_proj', default=False)
parser.add_argument('--q_proj', default=True)
parser.add_argument('--v_proj', default=True)
parser.add_argument('--r_proj', default=True)

parser.add_argument('--attn_ff', default=False)

# parser.add_argument('--rel_pos', default=False)
parser.add_argument('--use_abs_pos', default=False)
parser.add_argument('--use_rel_pos', default=True)
#相对位置和绝对位置不是对立的，可以同时使用
parser.add_argument('--rel_pos_shared', default=True)
parser.add_argument('--add_pos', default=False)
parser.add_argument('--learn_pos', default=False)
parser.add_argument('--pos_norm', default=False)
parser.add_argument('--rel_pos_init', default=1)
parser.add_argument('--four_pos_shared', default=True,help='只针对相对位置编码，指4个位置编码是不是共享权重')
parser.add_argument('--four_pos_fusion', default='ff_two',choices=['ff','attn','gate','ff_two','ff_linear'],
                    help='ff就是输入带非线性隐层的全连接，'
                         'attn就是先计算出对每个位置编码的加权，然后求加权和'
                         'gate和attn类似，只不过就是计算的加权多了一个维度')

parser.add_argument('--four_pos_fusion_shared', default=True, help='是不是要共享4个位置融合之后形成的pos')

# parser.add_argument('--rel_pos_scale',default=2,help='在lattice且用相对位置编码时，由于中间过程消耗显存过大，'
#                                                  '所以可以使4个位置的初始embedding size缩小，'
#                                                  '最后融合时回到正常的hidden size即可')

parser.add_argument('--pre', default='')
parser.add_argument('--post', default='an')

over_all_dropout =  -1
parser.add_argument('--embed_dropout_before_pos', default=False)
parser.add_argument('--embed_dropout', default=0.5,type=float)
parser.add_argument('--gaz_dropout', default=0.5,type=float)
parser.add_argument('--output_dropout', default=0.3,type=float)
parser.add_argument('--pre_dropout', default=0.5,type=float)
parser.add_argument('--post_dropout', default=0.3,type=float)
parser.add_argument('--ff_dropout', default=0.15,type=float)
parser.add_argument('--ff_dropout_2', default=-1,type=float,help='FF第二层过完后的dropout，之前没管这个的时候是0')
parser.add_argument('--attn_dropout', default=0,type=float)
parser.add_argument('--embed_dropout_pos', default='0')
parser.add_argument('--abs_pos_fusion_func', default='nonlinear_add',
                    choices=['add','concat','nonlinear_concat','nonlinear_add','concat_nonlinear','add_nonlinear'])

# For databaker, pre-tokenizer all information including (lexicon|rule|attribute)
parser.add_argument('--dataset', default='databaker', help='weibo|resume|ontonotes|msra|databaker(for TN task)')
# parser.add_argument('--debug',default=1)


args = parser.parse_args()
if args.ff_dropout_2 < 0:
    args.ff_dropout_2 = args.ff_dropout

if over_all_dropout > 0:
    args.embed_dropout = over_all_dropout
    args.output_dropout = over_all_dropout
    args.pre_dropout = over_all_dropout
    args.post_dropout = over_all_dropout
    args.ff_dropout = over_all_dropout
    args.attn_dropout = over_all_dropout

if args.lattice and args.use_rel_pos:
    args.train_clip = True

# fitlog.commit(__file__,fit_msg='绝对位置用新的了')
fitlog.set_log_dir('logs')
now_time = get_peking_time()
logger.add_file('log/{}'.format(now_time),level='info')
if args.test_batch == -1:
    args.test_batch = args.batch//2
fitlog.add_hyper(now_time,'time')
if args.debug:
    # args.dataset = 'toy'
    pass

if args.device != 'cpu':
    assert args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')

refresh_data = True

for k,v in args.__dict__.items():
    print_info('{}:{}'.format(k, v))

# if args.dataset == 'ontonote':
#     datasets, vocabs, embeddings = load_ontonotes4ner(ontonote4ner_cn_path, yangjie_rich_pretrain_unigram_path,
#                                                       yangjie_rich_pretrain_bigram_path,
#                                                       _refresh=refresh_data, index_token=True,
#                                                       )
# elif args.dataset == 'resume':
#     datasets, vocabs, embeddings = load_resume_ner(resume_ner_path, yangjie_rich_pretrain_unigram_path,
#                                                    yangjie_rich_pretrain_bigram_path,
#                                                    _refresh=refresh_data, index_token=True,
#                                                    )
# elif args.dataset == 'weibo':
#     datasets, vocabs, embeddings = load_weibo_ner(weibo_ner_path, yangjie_rich_pretrain_unigram_path,
#                                                   yangjie_rich_pretrain_bigram_path,
#                                                   _refresh=refresh_data, index_token=True,
#                                                   )
# elif args.dataset == 'weibo_old':
#     datasets, vocabs, embeddings = load_weibo_ner_old(weibo_ner_old_path, yangjie_rich_pretrain_unigram_path,
#                                                       yangjie_rich_pretrain_bigram_path,
#                                                       _refresh=refresh_data, index_token=True,
#                                                       )

# print(max(datasets['train']['seq_len']))
# print(max(datasets['dev']['seq_len']))
# print(max(datasets['test']['seq_len']))
# exit(0)

raw_dataset_cache_name = os.path.join('cache',args.dataset+
                          '_trainClip:{}'.format(args.train_clip)
                                      +'bgminfreq_{}'.format(args.bigram_min_freq)
                                      +'char_min_freq_{}'.format(args.char_min_freq)
                                      +'word_min_freq_{}'.format(args.word_min_freq)
                                      +'only_train_min_freq{}'.format(args.only_train_min_freq)
                                      +'number_norm{}'.format(args.number_normalized)
                                      + 'load_dataset_seed{}'.format(load_dataset_seed))


if args.dataset == 'ontonotes':
    datasets,vocabs,embeddings = load_ontonotes4ner(ontonote4ner_cn_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    _refresh=refresh_data,index_token=False,train_clip=args.train_clip,
                                                    _cache_fp=raw_dataset_cache_name,
                                                    char_min_freq=args.char_min_freq,
                                                    bigram_min_freq=args.bigram_min_freq,
                                                    only_train_min_freq=args.only_train_min_freq
                                                    )
elif args.dataset == 'resume':
    print('!!!!!!!!!!!!!!!!!!!!!! RESUME dataset works here.')
    datasets, vocabs, embeddings = load_resume_ner(resume_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    _refresh=refresh_data,index_token=False,
                                                    _cache_fp=raw_dataset_cache_name,
                                                    char_min_freq=args.char_min_freq,
                                                    bigram_min_freq=args.bigram_min_freq,
                                                    only_train_min_freq=args.only_train_min_freq
                                                    )
elif args.dataset == 'weibo':
    datasets,vocabs,embeddings = load_weibo_ner(weibo_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    _refresh=refresh_data,index_token=False,
                                                _cache_fp=raw_dataset_cache_name,
                                                char_min_freq=args.char_min_freq,
                                                bigram_min_freq=args.bigram_min_freq,
                                                only_train_min_freq=args.only_train_min_freq
                                                    )

elif args.dataset == 'toy':
    datasets,vocabs,embeddings = load_toy_ner(toy_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    _refresh=refresh_data,index_token=False,train_clip=args.train_clip,
                                                    _cache_fp=raw_dataset_cache_name
                                                    )

elif args.dataset == 'databaker':
    print('!!!!!!!!!!!!!!!!!!!!!! DATABAKER dataset works here.')
    datasets, vocabs, embeddings = load_databaker_tn(databaker_tn_path, yangjie_rich_pretrain_unigram_path, yangjie_rich_pretrain_bigram_path,
                                                _refresh=refresh_data, index_token=False,
                                                _cache_fp=raw_dataset_cache_name,
                                                char_min_freq=args.char_min_freq,
                                                bigram_min_freq=args.bigram_min_freq,
                                                only_train_min_freq=args.only_train_min_freq
                                                )


if args.gaz_dropout < 0:
    args.gaz_dropout = args.embed_dropout

args.hidden = args.head_dim * args.head
args.ff = args.hidden * args.ff


if args.dataset == 'weibo':
    pass
elif args.dataset == 'resume':
    pass
elif args.dataset == 'ontonotes':
    args.update_every = 2
    pass
elif args.dataset == 'msra':
    pass


if args.lexicon_name == 'lk':
    yangjie_rich_pretrain_word_path = lk_word_path_2

print('用的词表的路径:{}'.format(yangjie_rich_pretrain_word_path))

w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                              _refresh=refresh_data,
                                              _cache_fp='cache/{}'.format(args.lexicon_name))

cache_name = os.path.join('cache',(args.dataset+'_lattice'+'_only_train:{}'
                                    +'_trainClip:{}'+'_norm_num:{}'
                                    +'char_min_freq{}'+'bigram_min_freq{}'+'word_min_freq{}'+'only_train_min_freq{}'
                                    +'number_norm{}'+'lexicon_{}'+'load_dataset_seed_{}')
                            .format(args.only_lexicon_in_train,
                            args.train_clip,args.number_normalized,args.char_min_freq,
                            args.bigram_min_freq,args.word_min_freq,args.only_train_min_freq,
                            args.number_normalized,args.lexicon_name,load_dataset_seed))

## add lattice and rule HERE
if args.dataset == 'databaker':
    datasets, vocabs, embeddings = equip_chinese_tn_with_lexicon_and_rule(datasets, vocabs, embeddings,
                                                            w_list, yangjie_rich_pretrain_word_path,
                                                            _refresh=refresh_data, _cache_fp=cache_name,
                                                            only_lexicon_in_train=args.only_lexicon_in_train,
                                                            word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                            number_normalized=args.number_normalized,
                                                            lattice_min_freq=args.lattice_min_freq,
                                                            only_train_min_freq=args.only_train_min_freq,
                                                            use_lexicon=args.use_lexicon,
                                                            use_rule=args.use_rule)
else:
    datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets, vocabs, embeddings,
                                                            w_list, yangjie_rich_pretrain_word_path,
                                                            _refresh=refresh_data, _cache_fp=cache_name,
                                                            only_lexicon_in_train=args.only_lexicon_in_train,
                                                            word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                            number_normalized=args.number_normalized,
                                                            lattice_min_freq=args.lattice_min_freq,
                                                            only_train_min_freq=args.only_train_min_freq
                                                            )

print('train:{}'.format(len(datasets['train'])))
avg_seq_len = 0
avg_lex_num = 0
avg_seq_lex = 0
train_seq_lex = []
dev_seq_lex = []
test_seq_lex = []
train_seq = []
dev_seq = []
test_seq = []
for k,v in datasets.items():
    max_seq_len = 0
    max_lex_num = 0
    max_seq_lex = 0
    max_seq_len_i = -1
    for i in range(len(v)):
        if max_seq_len < v[i]['seq_len']:
            max_seq_len = v[i]['seq_len']
            max_seq_len_i = i
        # max_seq_len = max(max_seq_len,v[i]['seq_len'])
        max_lex_num = max(max_lex_num, v[i]['lex_num'])
        max_seq_lex = max(max_seq_lex, v[i]['lex_num'] + v[i]['seq_len'])

        avg_seq_len += v[i]['seq_len']
        avg_lex_num += v[i]['lex_num']
        avg_seq_lex += (v[i]['seq_len']+v[i]['lex_num'])
        if k == 'train':
            train_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
            train_seq.append(v[i]['seq_len'])
            if v[i]['seq_len'] > 200:
                print('train里这个句子char长度已经超了200了')
                print(''.join(list(map(lambda x:vocabs['char'].to_word(x),v[i]['chars']))))
            else:
                if v[i]['seq_len']+v[i]['lex_num'] > 400:
                    print('train里这个句子char长度没超200，但是总长度超了400')
                    print(''.join(list(map(lambda x: vocabs['char'].to_word(x), v[i]['chars']))))
        if k == 'dev':
            dev_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
            dev_seq.append(v[i]['seq_len'])
        if k == 'test':
            test_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
            test_seq.append(v[i]['seq_len'])


    print('{} 最长的句子是:{}'.format(k, list(map(lambda x:vocabs['char'].to_word(x), v[max_seq_len_i]['chars']))))
    print('{} max_seq_len:{}'.format(k, max_seq_len))
    print('{} max_lex_num:{}'.format(k, max_lex_num))
    print('{} max_seq_lex:{}'.format(k, max_seq_lex))

# exit(1208)

# pickle.dump(train_seq_lex,open('train_seq_lex','wb'))
# pickle.dump(dev_seq_lex,open('dev_seq_lex','wb'))
# pickle.dump(test_seq_lex,open('test_seq_lex','wb'))
#
# pickle.dump(train_seq,open('train_seq','wb'))
# pickle.dump(dev_seq,open('dev_seq','wb'))
# pickle.dump(test_seq,open('test_seq','wb'))
# exit(1208)

# avg_seq_len/=(len(datasets['train'])+len(datasets['dev'])+len(datasets['test']))
# avg_lex_num/=(len(datasets['train'])+len(datasets['dev'])+len(datasets['test']))
# avg_seq_lex/=(len(datasets['train'])+len(datasets['dev'])+len(datasets['test']))

# #画图开始
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib
# # matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
# # matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
#
# plt.hist(train_seq_lex, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
# # 显示横轴标签
# plt.xlabel("区间")
# # 显示纵轴标签
# plt.ylabel("频数/频率")
# # 显示图标题
# plt.title("频数/频率分布直方图")
# plt.show()
#
# print('avg_seq_len:{}'.format(avg_seq_len))
# print('avg_lex_num:{}'.format(avg_lex_num))
# print('avg_seq_lex:{}'.format(avg_seq_lex))
# exit(1208)


# max_seq_len = max(max(datasets['train']['seq_len']),max(datasets['dev']['seq_len']),max(datasets['test']['seq_len']))
# import copy
max_seq_len = max(* map(lambda x:max(x['seq_len']), datasets.values()))

show_index = 4
print('raw_chars:{}'.format(list(datasets['train'][show_index]['raw_chars'])))
print('lexicons:{}'.format(list(datasets['train'][show_index]['lexicons'])))
print('rules:{}'.format(list(datasets['train'][show_index]['rules'])))
print('lattice:{}'.format(list(datasets['train'][show_index]['lattice'])))
print('raw_lattice:{}'.format(list(map(lambda x:vocabs['lattice'].to_word(x),
                                  list(datasets['train'][show_index]['lattice'])))))
print('lex_s:{}'.format(list(datasets['train'][show_index]['lex_s'])))
print('lex_e:{}'.format(list(datasets['train'][show_index]['lex_e'])))
print('pos_s:{}'.format(list(datasets['train'][show_index]['pos_s'])))
print('pos_e:{}'.format(list(datasets['train'][show_index]['pos_e'])))

# exit(1208)

for k, v in datasets.items():
    # v.apply_field(lambda x:x,'chars',new_field_name='chars_target')
    # v.set_pad_val('chars_target',pad_val=-100)
    # print_info(v[0])
    # v.set_input('chars_target')
    # v.set_target('chars_target')
    if args.lattice:
        v.set_input('lattice', 'bigrams', 'seq_len', 'target')
        v.set_input('lex_num', 'pos_s', 'pos_e')
        v.set_target('target', 'seq_len')
        v.set_pad_val('lattice', vocabs['lattice'].padding_idx)
    else:
        v.set_input('chars', 'bigrams', 'seq_len', 'target')
        v.set_target('target', 'seq_len')


# print(embeddings['char'].embedding.weight[:10])
if args.norm_embed > 0:
    print('embedding:{}'.format(embeddings['char'].embedding.weight.size()))
    print('norm embedding')
    for k, v in embeddings.items():
        norm_static_embedding(v, args.norm_embed)

if args.norm_lattice_embed > 0:
    print('embedding:{}'.format(embeddings['lattice'].embedding.weight.size()))
    print('norm lattice embedding')
    for k,v in embeddings.items():
        norm_static_embedding(v,args.norm_embed)


#####################
### model prepare ###
#####################

mode = {}
mode['debug'] = args.debug
mode['gpumm'] = args.gpumm
if args.debug or args.gpumm:
    fitlog.debug()
dropout = collections.defaultdict(int)
dropout['embed'] = args.embed_dropout
dropout['gaz'] = args.gaz_dropout
dropout['output'] = args.output_dropout
dropout['pre'] = args.pre_dropout
dropout['post'] = args.post_dropout
dropout['ff'] = args.ff_dropout
dropout['ff_2'] = args.ff_dropout_2
dropout['attn'] = args.attn_dropout

torch.backends.cudnn.benchmark = False
fitlog.set_rng_seed(args.seed)
torch.backends.cudnn.benchmark = False
fitlog.add_hyper(args)


if args.model == 'transformer':
    if args.lattice:
        if args.use_bert:
            bert_embedding = BertEmbedding(vocabs['lattice'], model_dir_or_name='cn-wwm', 
                                            requires_grad=False, word_dropout=0.01, auto_truncate=True)
        else:
            bert_embedding = None

        if args.only_bert:

            print('CHECK BERT_SeqLabel works here!!!!!')
            print('CHECK BERT_SeqLabel args.after_bert: ', args.after_bert)

            model = BERT_SeqLabel(bert_embedding, len(vocabs['label']), vocabs, args.after_bert)
        else:

            print('CHECK Lattice_Transformer_SeqLabel works here!!!!!')

            model = Lattice_Transformer_SeqLabel(embeddings['lattice'], embeddings['bigram'], 
                                         args.hidden, 
                                         len(vocabs['label']),
                                         args.head, args.layer, args.use_abs_pos,args.use_rel_pos,
                                         args.learn_pos, args.add_pos,
                                         args.pre, args.post, args.ff, args.scaled, dropout, args.use_bigram,
                                         mode, device, vocabs,
                                         max_seq_len=max_seq_len,
                                         rel_pos_shared=args.rel_pos_shared,
                                         k_proj=args.k_proj,
                                         q_proj=args.q_proj,
                                         v_proj=args.v_proj,
                                         r_proj=args.r_proj,
                                         self_supervised=args.self_supervised,
                                         attn_ff=args.attn_ff,
                                         pos_norm=args.pos_norm,
                                         ff_activate=args.ff_activate,
                                         abs_pos_fusion_func=args.abs_pos_fusion_func,
                                         embed_dropout_pos=args.embed_dropout_pos,
                                         four_pos_shared=args.four_pos_shared,
                                         four_pos_fusion=args.four_pos_fusion,
                                         four_pos_fusion_shared=args.four_pos_fusion_shared,
                                         bert_embedding=bert_embedding,
                                         use_char_lstm=args.use_char_lstm,
                                         use_lexicon=args.use_lexicon,
                                         use_rule=args.use_rule
                                         )
    else:
        model = Transformer_SeqLabel(embeddings['lattice'], embeddings['bigram'], args.hidden, len(vocabs['label']),
                                     args.head, args.layer, args.use_abs_pos,args.use_rel_pos,
                                     args.learn_pos, args.add_pos,
                                     args.pre, args.post, args.ff, args.scaled,dropout,args.use_bigram,
                                     mode,device,vocabs,
                                     max_seq_len=max_seq_len,
                                     rel_pos_shared=args.rel_pos_shared,
                                     k_proj=args.k_proj,
                                     q_proj=args.q_proj,
                                     v_proj=args.v_proj,
                                     r_proj=args.r_proj,
                                     self_supervised=args.self_supervised,
                                     attn_ff=args.attn_ff,
                                     pos_norm=args.pos_norm,
                                     ff_activate=args.ff_activate,
                                     abs_pos_fusion_func=args.abs_pos_fusion_func,
                                     embed_dropout_pos=args.embed_dropout_pos
                                     )

    # print(Transformer_SeqLabel.encoder.)
elif args.model =='lstm':
    model = LSTM_SeqLabel_True(embeddings['char'],embeddings['bigram'],embeddings['bigram'],args.hidden,
                               len(vocabs['label']),
                          bidirectional=True,device=device,
                          embed_dropout=args.embed_dropout,output_dropout=args.output_dropout,use_bigram=True,
                          debug=args.debug)

for n,p in model.named_parameters():
    print('{}:{}'.format(n, p.size()))

# exit(1208)
# for k,v in model.state_dict().items():
#     # print(k,v)
#     print('{}:{}'.format(k,v.size()))
# exit(1208)
# for mod in model.modules():
#     print(mod)
# print('的:{}'.format(embeddings['char'](vocabs['char'].to_index('的'))))
# print('output layer:{}'.format(model.output.weight))
# print('这次让pytorch默认初始化transformer')

with torch.no_grad():
    print_info('{}init pram{}'.format('*'*15,'*'*15))
    for n,p in model.named_parameters():
        if 'bert' not in n and 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                and 'bias' not in n and 'crf' not in n and p.dim()>1:
            try:
                if args.init == 'uniform':
                    nn.init.xavier_uniform_(p)
                    print_info('xavier uniform init:{}'.format(n))
                elif args.init == 'norm':
                    print_info('xavier norm init:{}'.format(n))
                    nn.init.xavier_normal_(p)
            except:
                print_info(n)
                exit(1208)
    print_info('{}init pram{}'.format('*' * 15, '*' * 15))

loss = LossInForward()
# loss = CrossEntropyLoss()
encoding_type = 'bmeso'
if args.dataset == 'weibo':
    encoding_type = 'bio'
# f1_metric = SpanFPreRecMetric(vocabs['label'], pred='pred', target='target', seq_len='seq_len', encoding_type=encoding_type)
f1_metric = SpanFPreRecMetric(vocabs['label'], pred='pred', target='target', seq_len='seq_len', encoding_type=encoding_type, only_gross=False)
acc_metric = AccuracyMetric(pred='pred', target='target', seq_len='seq_len', )
acc_metric.set_metric_name('label_acc')
metrics = [
    f1_metric,
    acc_metric
]
if args.self_supervised:
    chars_acc_metric = AccuracyMetric(pred='chars_pred', target='chars_target', seq_len='seq_len')
    chars_acc_metric.set_metric_name('chars_acc')
    metrics.append(chars_acc_metric)

if args.see_param:
    for n, p in model.named_parameters():
        print_info('{}:{}'.format(n,p.size()))
    print_info('see_param mode: finish')
    if not args.debug:
        exit(1208)

datasets['train'].apply


if args.see_convergence:   # default False
    print_info('see_convergence = True')
    print_info('so just test train acc|f1')
    datasets['train'] = datasets['train'][:100]
    if args.optim == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # declare a trainer with all parameters of model to see convergence?
    trainer = Trainer(datasets['train'], model, optimizer, loss, args.batch,
                      n_epochs=args.epoch, dev_data=datasets['train'], metrics=metrics,
                      device=device, dev_batch_size=args.test_batch)

    trainer.train()
    exit(1208)


if not args.only_bert:
    if not args.use_bert:
        bigram_embedding_param = list(model.bigram_embed.parameters())
        gaz_embedding_param = list(model.lattice_embed.parameters())
        embedding_param = bigram_embedding_param
        if args.lattice:
            gaz_embedding_param = list(model.lattice_embed.parameters())
            embedding_param = embedding_param+gaz_embedding_param
        embedding_param_ids = list(map(id,embedding_param))
        non_embedding_param = list(filter(lambda x:id(x) not in embedding_param_ids,model.parameters()))
        param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate}]
    else:
        bert_embedding_param = list(model.bert_embedding.parameters())
        bert_embedding_param_ids = list(map(id,bert_embedding_param))
        bigram_embedding_param = list(model.bigram_embed.parameters())
        gaz_embedding_param = list(model.lattice_embed.parameters())
        embedding_param = bigram_embedding_param
        if args.lattice:
            gaz_embedding_param = list(model.lattice_embed.parameters())
            embedding_param = embedding_param+gaz_embedding_param
        embedding_param_ids = list(map(id,embedding_param))
        non_embedding_param = list(filter(
            lambda x:id(x) not in embedding_param_ids and id(x) not in bert_embedding_param_ids,
                                          model.parameters()))
        param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate},
                  {'params':bert_embedding_param,'lr':args.bert_lr_rate*args.lr}]
else:
    non_embedding_param = model.parameters()
    embedding_param = []
    param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate}]


if args.optim == 'adam':
    optimizer = optim.AdamW(param_, lr=args.lr, weight_decay=args.weight_decay)
elif args.optim == 'sgd':
    optimizer = optim.SGD(param_, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)


if args.dataset == 'msra':
    datasets['dev']  = datasets['test']
fitlog_evaluate_dataset = {'test': datasets['test']}
if args.test_train:
    fitlog_evaluate_dataset['train'] = datasets['train']
evaluate_callback = FitlogCallback(fitlog_evaluate_dataset, verbose=1)
lrschedule_callback = LRScheduler(lr_scheduler=LambdaLR(optimizer, lambda ep: 1 / (1 + 0.05*ep) ))
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)


class Unfreeze_Callback(Callback):
    def __init__(self,bert_embedding,fix_epoch_num):
        super().__init__()
        self.bert_embedding = bert_embedding
        self.fix_epoch_num = fix_epoch_num
        assert self.bert_embedding.requires_grad == False

    def on_epoch_begin(self):
        if self.epoch == self.fix_epoch_num+1:
            self.bert_embedding.requires_grad = True

callbacks = [
        evaluate_callback,
        lrschedule_callback,
        clip_callback
    ]

if args.use_bert:
    if args.fix_bert_epoch != 0:
        callbacks.append(Unfreeze_Callback(bert_embedding, args.fix_bert_epoch))
    else:
        bert_embedding.requires_grad = True
callbacks.append(EarlyStopCallback(args.early_stop))
if args.warmup > 0 and args.model == 'transformer':
    callbacks.append(WarmupCallback(warmup=args.warmup))


class record_best_test_callback(Callback):
    def __init__(self,trainer,result_dict):
        super().__init__()
        self.trainer222 = trainer
        self.result_dict = result_dict

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        print(eval_result['data_test']['SpanFPreRecMetric']['f'])

# check the device during running
print(torch.rand(size=[3,3], device=device))

# begin to train HERE!
if args.status == 'train':
    trainer = Trainer(datasets['train'], model, optimizer, loss, args.batch,
                      n_epochs=args.epoch,
                      dev_data=datasets['dev'],
                      metrics=metrics,
                      device=device, callbacks=callbacks, dev_batch_size=args.test_batch,
                      test_use_tqdm=False, check_code_level=-1, # validate_every=100,
                      update_every=args.update_every)

    trainer.train()
