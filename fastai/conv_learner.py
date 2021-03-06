from .core import *
from .layers import *
from .learner import *
from .initializers import *

model_meta = {
    resnet18:[8,6], resnet34:[8,6], resnet50:[8,6], resnet101:[8,6], resnet152:[8,6],
    vgg16:[0,22], vgg19:[0,22],
    resnext50:[8,6], resnext101:[8,6], resnext101_64:[8,6],
    wrn:[8,6], inceptionresnet_2:[-2,9], inception_4:[-1,9],
    dn121:[0,7], dn161:[0,7], dn169:[0,7], dn201:[0,7],
}
model_features = {inception_4: 3072, dn121: 2048, dn161: 4416,} # nasnetalarge: 4032*2}

class ConvnetBuilder():
    """Class representing a convolutional network.

    Arguments:
        model_creation_function: a model creation function (e.g. resnet34, vgg16, etc)
        last_layer_size (int): size of the last layer
        is_multi (bool): is multilabel classification?
            (def here http://scikit-learn.org/stable/modules/multiclass.html)
        is_reg (bool): is a regression?
        dropout_percs (float or array of float): dropout parameters
        xtra_fc (list of ints): list of hidden layers with # hidden neurons
        xtra_cut (int): # layers earlier than default to cut the model, detault is 0
    """

    def __init__(self, model_creation_function, last_layer_size, is_multi, is_reg, dropout_percs=None, xtra_fc=None, xtra_cut=0):
        self.model_creation_function,self.last_layer_size,self.is_multi,self.is_reg,self.xtra_cut = model_creation_function,last_layer_size,is_multi,is_reg,xtra_cut
        if dropout_percs is None: dropout_percs = [0.25,0.5]
        if xtra_fc is None: xtra_fc = [512]
        self.dropout_percs,self.xtra_fc = dropout_percs,xtra_fc

        if model_creation_function in model_meta: cut,self.lr_cut = model_meta[model_creation_function]
        else: cut,self.lr_cut = 0,0
        cut-=xtra_cut
        layers = cut_model(model_creation_function(True), cut)
        self.nf = model_features[model_creation_function] if model_creation_function in model_features else (num_features(layers)*2)
        layers += [AdaptiveConcatPool2d(), Flatten()]
        self.top_model = nn.Sequential(*layers)

        n_fc = len(self.xtra_fc)+1
        if not isinstance(self.dropout_percs, list): self.dropout_percs = [self.dropout_percs]*n_fc

        fc_layers = self.get_fc_layers()
        self.n_fc = len(fc_layers)
        self.fc_model = to_gpu(nn.Sequential(*fc_layers))
        apply_init(self.fc_model, kaiming_normal)
        self.model = to_gpu(nn.Sequential(*(layers+fc_layers)))

    @property
    def name(self): return f'{self.model_creation_function.__name__}_{self.xtra_cut}'

    def create_fc_layer(self, ni, nf, p, actn=None):
        res=[nn.BatchNorm1d(num_features=ni)]
        if p: res.append(nn.Dropout(p=p))
        res.append(nn.Linear(in_features=ni, out_features=nf))
        if actn: res.append(actn)
        return res

    def get_fc_layers(self):
        res=[]
        ni=self.nf
        for i,nf in enumerate(self.xtra_fc):
            res += self.create_fc_layer(ni, nf, p=self.dropout_percs[i], actn=nn.ReLU())
            ni=nf
        final_actn = nn.Sigmoid() if self.is_multi else nn.LogSoftmax()
        if self.is_reg: final_actn = None
        res += self.create_fc_layer(ni, self.last_layer_size, p=self.dropout_percs[-1], actn=final_actn)
        return res

    def get_layer_groups(self, do_fc=False):
        if do_fc:
            return [self.fc_model]
        idxs = [self.lr_cut]
        c = children(self.top_model)
        if len(c)==3: c = children(c[0])+c[1:]
        lgs = list(split_by_idxs(c,idxs))
        return lgs+[self.fc_model]


class ConvLearner(Learner):
    def __init__(self, data, models, precompute=False, **kwargs):
        self.precompute = False
        super().__init__(data, models, **kwargs)
        self.crit = F.binary_cross_entropy if data.is_multi else F.nll_loss
        if data.is_reg: self.crit = F.l1_loss
        elif self.metrics is None:
            self.metrics = [accuracy_thresh(0.5)] if self.data.is_multi else [accuracy]
        if precompute: self.save_fc1()
        self.freeze()
        self.precompute = precompute

    @classmethod
    def pretrained(cls, model_constructor, data, dropout_percs=None, xtra_fc=None, xtra_cut=0, precompute=True, **kwargs):
        """
        Creates a pretrained model
        
        Arguments:
          model_constructor: a model creation function (e.g. resnet34, vgg16, etc)
          dropout_percs: percentage of weights to set to 0 (dropout)
          precompute: whether the precomputed activations should be used
        """
        models = ConvnetBuilder(model_constructor, data.c, data.is_multi, data.is_reg, dropout_percs=dropout_percs, xtra_fc=xtra_fc, xtra_cut=xtra_cut)
        return cls(data, models, precompute, **kwargs)

    @property
    def model(self): return self.models.fc_model if self.precompute else self.models.model

    @property
    def data(self): return self.fc_data if self.precompute else self.data_

    def create_empty_bcolz(self, n, name):
        return bcolz.carray(np.zeros((0,n), np.float32), chunklen=1, mode='w', rootdir=name)

    def set_data(self, data, precompute=False):
        super().set_data(data)
        if precompute:
            self.unfreeze()
            self.save_fc1()
            self.freeze()
            self.precompute = True
        else:
            self.freeze()

    def get_layer_groups(self):
        return self.models.get_layer_groups(self.precompute)

    def summary(self):
        temp = self.precompute
        self.precompute = False
        res = super().summary()
        self.precompute = temp 
        return res

    def get_activations(self, force=False):
        tmpl = f'_{self.models.name}_{self.data.sz}.bc'
        # TODO: Somehow check that directory names haven't changed (e.g. added test set)
        names = [os.path.join(self.tmp_path, p+tmpl) for p in ('x_act', 'x_act_val', 'x_act_test')]
        if os.path.exists(names[0]) and not force:
            self.activations = [bcolz.open(p) for p in names]
        else:
            self.activations = [self.create_empty_bcolz(self.models.nf,n) for n in names]

    def save_fc1(self):
        self.get_activations()
        activations, val_activations, test_activations = self.activations
        m=self.models.top_model
        if len(self.activations[0])!=len(self.data.train_ds):
            predict_to_bcolz(m, self.data.fix_dl, activations)
        if len(self.activations[1])!=len(self.data.val_ds):
            predict_to_bcolz(m, self.data.val_dl, val_activations)
        if self.data.test_dl and (len(self.activations[2])!=len(self.data.test_ds)):
            if self.data.test_dl: predict_to_bcolz(m, self.data.test_dl, test_activations)

        self.fc_data = ImageClassifierData.from_arrays(self.data.path,
                (activations, self.data.train_y), (val_activations, self.data.val_y), self.data.bs, classes=self.data.classes,
                test = test_activations if self.data.test_dl else None, num_workers=8)

    def freeze(self):
        """ Freeze all but the very last layer.

        Make all layers untrainable (i.e. frozen) except for the last layer.

        Returns:
            None
        """
        self.freeze_to(-1)

    def unfreeze(self):
        """ Unfreeze all layers.

        Make all layers trainable by unfreezing. This will also set the `precompute` to `False` since we can
        no longer pre-calculate the activation of frozen layers.

        Returns:
            None
        """
        self.freeze_to(0)
        self.precompute = False
