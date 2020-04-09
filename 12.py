'''
transformeer 使用

'''

import torch
print(torch.__version__)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 下面的库包有,只是pycharm没找到而已.
from torch.nn import TransformerEncoder, TransformerEncoderLayer
'''

C:/Users/Administrator/AppData/Local/Programs/Python/Python36/Lib/site-packages/torch/nn/modules/transformer.py



'''



print(TransformerEncoder)
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None

        # 核心是这3个层.
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)



        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

# 一定要遮挡future的数据,否则就学的就不真实了.
    def _generate_square_subsequent_mask(self, sz):
        # triu: 上三角矩阵.
        # 首先做一个sz,sz shape 全一张亮.然后zhuanzhi.
        # 这样这个mask*向量x 就会得到向量x之前的分量,不会得到future 的分量.保证逻辑正确.
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # mask_fill 用法https://blog.csdn.net/candy134834/article/details/84594754
        # 所以下面就是把==0的地方都换成-inf  ==1的地方换成0.
        # .float 会吧ture 和false 变成1 和0.
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        #如果灭有给src_mask 或者给的长度不对.那么就会出发.生成一个能对应上长度的给这个对象的self.srcnmask
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp) # 先进行嵌入编码
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output






class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # 写到公共变量里面方便使用.

    def forward(self, x):
        # 用加法加进去而已.
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 数据集的处理
import torchtext
from torchtext.data.utils import get_tokenizer # get_tokenizer 就是一个字典.这里面用的是
# 基础英语的字典.
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 吧数据分割成每一个batch. 比如26个英文字母就变成4个batch 每一个batch里面的元素是6个字母,最后2个字母扔了.
def batchify(data, bsz): # 数据在data.examples[0].text 这个里面.
    data = TEXT.numericalize([data.examples[0].text]) # 向量化.  # data: 3,12,3852
    # Divide the dataset into bsz parts.// data整个是一个大的段落.里面每一个数字代表一个单词的index  比如这里面就是3,12 分别对应eos 和= 这两个符号.  bsz 表示一行话只能20个字符.所以下面我们计算有多少个batch : nbatch
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz) #提取出整句,扔了后面没用的
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous() # reshape一下. 再.t是转置.
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)





bptt = 35 # 每35个算一句话
def get_batch(source, i):# 做数据和label
    seq_len = min(bptt, len(source) - 1 - i) # 表示剩余的句子跟bptt取min,得到应该取多长
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)#向后都偏移一个单位即可.
    return data, target



ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)





criterion = nn.CrossEntropyLoss() # 分类问题交叉熵就行了.
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)# 防止梯度爆炸
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)






# 运行上面的train和eval函数
# 先进行train ,和evaluate 训练之后 进行validate
best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

# 运行valudate
test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)




