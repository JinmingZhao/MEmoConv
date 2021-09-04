
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from codes.uniwav2vec_finetune.models.base_model import BaseModel
from codes.uniwav2vec_finetune.models.classifier import FcClassifier

class Wav2VecDNNModel(BaseModel):
    '''
    A: DNN
    V: denseface + LSTM + maxpool
    L: bert + textcnn
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--output_dim', type=int, default=7)
        parser.add_argument('--cls_layers', type=str, default=None)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--embd_method', type=str, default='last')
        parser.add_argument('--bidirection', action='store_true')
        parser.add_argument('--wav2vec_name', type=str, default='facebook/wav2vec2-base', help='which cross validation set')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE']
        self.model_names = ['enc', 'C']
        self.pretrained_model = ['enc']
        self.netenc = Wav2Vec2Model.from_pretrained(opt.wav2vec_name)
        if 'large' in opt.wav2vec_name:
            feature_dim = 1024
        else:
            feature_dim = 768
        if opt.cls_layers is not None:
            cls_layers = [int(l) for l in opt.cls_layers.split(',')]
        else:
            cls_layers = []
        self.netC = FcClassifier(feature_dim, cls_layers, opt.output_dim, dropout=0.3)
        
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.998)) # 0.999
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.signal = input['A_feat'].to(self.device)
        self.label = input['label'].to(self.device)
        self.input = input

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.segments = self.netenc(self.signal).last_hidden_state
        # print('before {}'.format(self.segments.shape))
        if self.opt.embd_method == 'last':
            self.segments = self.segments[:, -1]
        elif self.opt.embd_method == 'avg':
            self.segments = torch.mean(self.segments, axis=1)
        else:
            print('Error of the embedding method')
        # print('after {}'.format(self.segments.shape))
        self.logits, _ = self.netC(self.segments)
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        loss = self.loss_CE
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5.0) # 0.1

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()   
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 