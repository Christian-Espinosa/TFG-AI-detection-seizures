import torch


from eeg_models_Deb import *
from eeg_helpers import *
from eeg_dataset import *
from eeg_util_data import *


# from cvc_eeglib.eeg_utilitarios import *
#from pytorch_model_summary import summary
#def print_model_summary(model, input_shape=(14, 40)):
#    input_shape = (1,) + input_shape # change here to provide a sample of your dataset in the form [B, ...]
#    if next(model.parameters()).is_cuda:
#        print(summary(model, torch.randn(input_shape).cuda()))
#    else:
#        print(summary(model, torch.randn(input_shape)))

def instance_model(MODEL_NAME, WIN_SIZE, n_classes, lr, step_size=0):

    if MODEL_NAME == 'NN':
        # single neural network
        model = Seq_NN(n_electrodes=14, timesteps= WIN_SIZE * 8 , n_classes=n_classes).cuda()
        # print_model_summary(model)
    elif MODEL_NAME =='NNChannel':
        model = Seq_NN_Channels(n_channels=5,n_electrodes=14,
                                timesteps= WIN_SIZE * 8 , n_classes=n_classes).cuda()
    elif MODEL_NAME == 'NNEns':
        # ensemble taking in account 14 nodes
        model = Seq_NN_Ensemble_Elias( timesteps= WIN_SIZE * 8, n_classes=n_classes,sel_nodes=None).cuda()
        # model = Seq_NN_Ensemble_DifOut( timesteps= WIN_SIZE * 8,
        #                                 n_classes=n_classes, comb_type=1).cuda()
    elif MODEL_NAME == 'NNEnsFeat':
        # ensemble taking in account 14 nodes
        model = Seq_NN_EnsembleFeatures( timesteps= WIN_SIZE * 8, n_classes=n_classes).cuda()

    elif MODEL_NAME == 'C1D' :
        model = Seq_C1D(n_classes=n_classes).cuda()
        # model = Seq_C1D_3C(n_classes=n_classes).cuda()
        # print_model_summary(model)
    elif MODEL_NAME =='C1D_vrs2':
        model = Seq_C1D_vrs2(n_classes=n_classes).cuda()
          
    elif MODEL_NAME == 'C1DEns':
        # ensemble taking in account 14 nodes
        model = Seq_C1D_Ensemble(n_classes=n_classes).cuda()
        # model = Seq_C1D_Ensemble_DifOut(n_classes=n_classes, comb_type=4).cuda()
        # print_model_summary(model)

    elif MODEL_NAME == 'C1DEns10':
        # ensemble taking in account 14 nodes
        model = Seq_C1D_Ensemble(n_classes=n_classes, sel_nodes=1).cuda()
        # model = Seq_C1D_Ensemble_DifOut(n_classes=n_classes, comb_type=4).cuda()
        # print_model_summary(model)

    elif MODEL_NAME == 'C2D':
        model = Seq_C2D(n_nodes=14, n_classes=n_classes).cuda()
        # print_model_summary(model)

    elif MODEL_NAME == 'C2D_TS':
        model = Seq_C2D_TS(n_nodes=14, n_classes=n_classes).cuda()
        # print_model_summary(model)

    elif MODEL_NAME == 'C2D_ST':
        model = Seq_C2D_ST(n_nodes=14, n_classes=n_classes).cuda()
        # print_model_summary(model)

    elif MODEL_NAME == 'C2DSpatTem1':
        model = Seq_C2D_SpatTemp_v1(n_nodes=14, n_classes=n_classes).cuda()
        # print_model_summary(model)

    elif MODEL_NAME == 'C2DSpatTem2':
        model = Seq_C2D_SpatTemp_v2(n_nodes=14, n_classes=n_classes).cuda()
        # print_model_summary(model)
        
    elif MODEL_NAME == 'C1DLSTM' :
        model = Seq_C1D_LSTM(n_nodes=14, n_classes=n_classes).cuda()        
        # print_model_summary(model)
        
    elif MODEL_NAME == 'LSTM' :
        model = Seq_LSTM(n_nodes=14, n_classes=n_classes).cuda()        
        print_model_summary(model)

    else:
        print('The selected model does not match!')
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)


    if step_size <= 0:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

    return model, optimizer, scheduler

def train_Deb(x_train, y_train, n_classes, batch_size, n_epochs,
                lr, step_size, transf, x_valid, y_valid, MODEL_NAME, WIN_SIZE, pth_full_name=None):

    train_dataloader = create_test_dataloader(x_train, y_train, transf, batch_size)

    if x_valid is not None and y_valid is not None:
        print('validation set', x_valid.shape[0])
        valid_dataloader = create_test_dataloader(x_valid, y_valid, transf, batch_size, shuffle=False)
        
    else:
        valid_dataset = None
        valid_dataloader = None

    # select the model
    sample_counts = np.array(class_sample_count(list(y_train)))
    classes_weight=1./sample_counts
    classes_weight=classes_weight/np.sum(classes_weight)
    classes_weight=torch.tensor(classes_weight, dtype=torch.float).cuda()
    criterion = nn.CrossEntropyLoss(weight=classes_weight)
    
    model, optimizer, scheduler = instance_model(MODEL_NAME, WIN_SIZE, n_classes,lr, step_size)


    # in the case that valid_dataloader is None, no model is saved by epochs
    model, avg_cost = train_model(model, optimizer, criterion, n_epochs,
                            train_dataloader, valid_dataloader,
                            verbose=False,
                            pth_full_name=pth_full_name,
                            best_val_loss=None, scheduler=None)

    return model, optimizer, avg_cost

def train(x_train, y_train, n_classes, batch_size, n_epochs,
                lr, step_size, transf, x_valid, y_valid, MODEL_NAME, WIN_SIZE, pth_full_name=None):

    train_dataloader = create_train_dataloader_unbalanced(x_train, y_train, transf, batch_size)

    if x_valid is not None and y_valid is not None:
        print('validation set', x_valid.shape[0])
        valid_dataloader = create_test_dataloader(x_valid, y_valid, transf, batch_size, shuffle=False)
        
    else:
        valid_dataset = None
        valid_dataloader = None

    # select the model
    criterion = nn.CrossEntropyLoss()
    model, optimizer, scheduler = instance_model(MODEL_NAME, WIN_SIZE, n_classes,lr, step_size)


    # in the case that valid_dataloader is None, no model is saved by epochs
    model, avg_cost = train_model(model, optimizer, criterion, n_epochs,
                            train_dataloader, valid_dataloader,
                            verbose=False,
                            pth_full_name=pth_full_name,
                            best_val_loss=None, scheduler=None)

    return model, optimizer, avg_cost


def create_test_dataloader(x_test, y_test, transf=False, batch_size=128, shuffle=False):
    if y_test.ndim > 1:
        print('Data was no properly encoded. Error in test_dataloader!')
        return

    if y_test.shape[0] < batch_size:
        batch_size = y_test.shape[0]

    test_dataset = EEG_Dataset(x_test, y_test, transf)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False)
    return test_dataloader


def create_train_dataloader_unbalanced(x_train, y_train, transf=False, batch_size=128):
    if y_train.ndim > 1:
        print('Data was no properly encoded. Error in train_dataloader!')
        return

    sample_counts = class_sample_count(list(y_train))
    classes_weight = 1. / torch.tensor(sample_counts, dtype=torch.float)
    samples_weight = torch.tensor([classes_weight[w] for w in y_train])

    # traind dataloader
    train_dataset = EEG_Dataset(x_train, y_train, transf)
    print('training set ', x_train.shape[0])

    # pytorch function for sampling batch based on weights or probabilities for each
    # element. To obtain a relative balaced batch, it uses replacement by default
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=False, sampler=sampler)
    return train_dataloader

# =============================================================================
#
# =============================================================================
def class_sample_count(labels):
    tags = set(labels) # unique categories
    my_dic = {i:labels.count(i) for i in tags}
    my_dic = dict(sorted(my_dic.items())) # weight should be ordered for the optimizer
    print(my_dic)
    samples = list(my_dic.values())
    return samples

def class_weight(labels):
    tags = set(labels) # unique categories
    my_dic = {i:labels.count(i) for i in tags}
    my_dic = dict(sorted(my_dic.items())) # weight should be ordered for the optimizer
    print(my_dic)
    max_val = max(my_dic.values())
    weights = [ round(max_val / my_dic[i], 2)  for i in my_dic.keys()]
    #weights = [ 1. / my_dic[i]  for i in my_dic.keys()]
    return weights

def summarize_labels(labels):
    tags = set(labels) # unique categories
    my_dict = {i:labels.count(i) for i in tags}
    for keys in my_dict.keys():
        print('{0} \t {1} \t {2:4.2f}'.format(keys, my_dict[keys], my_dict[keys]/len(labels)))
    print('Total: \t ', len(labels))

# =============================================================================
#
# =============================================================================
# saving and loading checkpoint mechanisms
def save_checkpoint(model, optimizer, val_loss, epoch, save_full_path):
    """
        Save a model and its weights.
    """
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss,
                  'epoch': epoch}
    torch.save(state_dict, save_full_path)
    print(f'\tSaved model at epoch {epoch}  ==> {save_full_path}')

def load_checkpoint(model, optimizer, save_full_path):
    """
        Load a model and its weights.
    """
    state_dict = torch.load(save_full_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    epoch = state_dict['epoch']
    print(f'Loaded model weights <== {save_full_path}')
    return epoch, val_loss

# saving and loading checkpoint mechanisms
def save_full_checkpoint(model, optimizer, save_full_path, metadata=None):
    """
        Save a model and its weights.
    """
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'metadata': metadata}
    torch.save(state_dict, save_full_path)
    print(f'\tSaved model weights at  ==> {save_full_path}')

def load_full_checkpoint(model, optimizer, save_full_path):
    """
        Load a model and its weights.
    """
    state_dict = torch.load(save_full_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    metadata = state_dict['metadata']
    print(f'Loaded model <== {save_full_path}')
    return metadata

# saving the model
def save_full_model(model, save_full_path, ):
    """
        Save the entire model
    """    
    torch.save(model, save_full_path)
    print(f'\tEntire model saved at  ==> {save_full_path}')
    
# loading the model
def load_full_model(save_full_path):
    """
        Load the entire model
    """    
    model = torch.load(save_full_path)
    print(f'\tEntire model loaded  <== {save_full_path}')
    
    return model


# =============================================================================
# https://blog.paperspace.com/pytorch-101-advanced/
# =============================================================================

# Modules
# Children
# Different learnig rate for layers


def plot_model_initialization(model):
    """
        model = Seq_C1D_LSTM()
        
    """
    for module in model.modules():
      if isinstance(module, nn.Conv1d):
        weights = module.weight
        weights = weights.reshape(-1).detach().cpu().numpy()
        print(module.bias)                                       # Bias to zero
        plt.hist(weights)
        plt.show()
