import sys
import torch
import optuna
import numpy as np
import torch.nn as nn
import dgl.function as fn
import numpy_indexed as npi
import torch.nn.functional as F

from tqdm import tqdm

from gensit.config import Config
from gensit.utils.exceptions import *
from gensit.utils.misc_utils import setup_logger, comp_deg_norm
from gensit.static.global_variables import OPTIMIZERS


# Store hyperparameters
DEFAULT_HYPERPARAMS = {
    "nodes_per_layer": 128,
    "num_hidden_layers": 1,
    "dropout": 0,
    "reg_param": 0,
    "negative_sampling_rate": 0,
    "grad_norm": 1.0,
    "optimizer": 'Adam',
    "learning_rate": +1e-5,
    "multitask_weights": [0.5, 0.25, 0.25]
}
MODEL_TYPE = 'graph_attention_network'
MODEL_PREFIX = 'gat_'

class GAT_Model(nn.Module):

    def __init__(
        self, 
        trial: optuna.trial,
        config: Config,
        graph, 
        n_features,
        dims: dict, 
        device:str = 'cpu', 
        **kwargs
    ):
        '''
        Inputs:
        ---------------------
        graph: the graph
        n_features: number of geographical features at origin/destination
        input_size: original node attributes' dimension
        nodes_per_layer: node embedding dimension
        num_hidden_layers: number of hidden layers in graph neural network
        dropout: dropout rate
        reg_param: regularization loss coefficient
        device: device

        Output:
        ---------------------
        embedding of nodes.

        To train the model, use get_loss() to get the overall loss.
        '''
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            console_level = level,
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevels( console_level = level )
        
        # Config file
        self.config = config
        # Device
        self.device = device
        # Trial 
        self.trial = trial
        # Type of learning model
        self.model_type = MODEL_TYPE
        self.model_prefix = MODEL_PREFIX
        # Graph
        self.graph = graph
        # tqdm flag
        self.tqdm_disabled = self.config['graph_attention_network'].get('disable_tqdm',True)

        # init super class
        super().__init__()
        
        # Update hyperparameters
        self.update_hyperparameters()

        # create modules
        # GAT for origin nodes
        self.origin_gat = GAT(
            graph = graph,
            num_regions = dims['origin'],
            input_size = n_features,
            nodes_per_layer = self.hyperparams['nodes_per_layer'], 
            output_size = self.hyperparams['nodes_per_layer'], 
            num_hidden_layers = self.hyperparams['num_hidden_layers'], 
            dropout = self.hyperparams['dropout'], 
            device = device
        ) 
        # GAT for destination nodes
        self.destination_gat = GAT(
            graph = graph,
            num_regions = dims['destination'], 
            input_size = n_features, 
            nodes_per_layer = self.hyperparams['nodes_per_layer'], 
            output_size = self.hyperparams['nodes_per_layer'], 
            num_hidden_layers = self.hyperparams['num_hidden_layers'], 
            dropout = self.hyperparams['dropout'], 
            device = device
        )
        # linear plan
        self.edge_regressor = nn.Bilinear(self.hyperparams['nodes_per_layer'], self.hyperparams['nodes_per_layer'], 1)
        self.in_regressor = nn.Linear(self.hyperparams['nodes_per_layer'], 1)
        self.out_regressor = nn.Linear(self.hyperparams['nodes_per_layer'], 1)
        # FNN plan
        # self.edge_regressor = FNN(nodes_per_layer * 2, dropout, device)
        # self.in_regressor = FNN(nodes_per_layer, dropout, device)
        # self.out_regressor = FNN(nodes_per_layer, dropout, device)
        
        # create a optimizer
        optimizer_function = eval(
            OPTIMIZERS[self.hyperparams['optimizer']],
            {"torch":torch}
        )
        self.optimizer = optimizer_function(self.parameters(), lr = self.hyperparams['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 100, gamma = 0.1)

    def update_hyperparameters(self):
        # Set hyperparams
        self.hyperparams = {}
        if self.trial is not None:
            OPTUNA_HYPERPARAMS = {
                # "optimizer":  self.trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                "nodes_per_layer": self.trial.suggest_categorical('nodes_per_layer', [32, 64, 128]),
                "num_hidden_layers": self.trial.suggest_int('num_hidden_layers', 0, 12, step = 1),
                "reg_param": self.trial.suggest_float('learning_rate', 0.0, 1.0),
                "learning_rate": self.trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
                "multitask_weights": [0.5, 0.25, 0.25]
            } 
        
        for pname in DEFAULT_HYPERPARAMS.keys():
            if self.trial is not None and pname in OPTUNA_HYPERPARAMS:
                if pname == 'multitask_weights':
                    # Sample from Dirichlet allocation
                    n = len(OPTUNA_HYPERPARAMS[pname])
                    w = []
                    for i in range(n):
                        w.append(- np.log(self.trial.suggest_float(f"w_{i}", 0, 1)))

                    p = []
                    for i in range(n):
                        p.append(w[i] / sum(w))

                    for i in range(n):
                        self.trial.set_user_attr(f"p_{i}", p[i])
                
                    self.hyperparams[pname] = p
                else:
                    self.hyperparams[pname] = OPTUNA_HYPERPARAMS[pname]
            
            elif self.config is None or getattr(self.config[self.model_type]['hyperparameters'],pname,None) is None:
                self.hyperparams[pname] = DEFAULT_HYPERPARAMS[pname]
            else:
                self.hyperparams[pname] =  self.config[self.model_type]['hyperparameters'][(self.model_prefix+pname)]

            if self.config is not None and getattr(self.config[self.model_type]['hyperparameters'],pname,None) is None:
                # Update object and config hyperparameters
                self.config[self.model_type]['hyperparameters'][(self.model_prefix+pname)] = self.hyperparams[pname]
            # Update batch size
            self.hyperparams['batch_size'] = self.config['training']['batch_size']

    def origin_forward(self, g):
        '''
        forward propagate of the graph to get embeddings for the origin node
        '''
        return self.origin_gat.forward(g)
    
    def destination_forward(self,g):
        '''
        forward propagate of the graph to get embeddings for destination node
        '''
        return self.destination_gat.forward(g)

    def get_loss(self, trip_od, scaled_trip_volume, in_flows, out_flows):
        '''
        defines the procedure of evaluating loss function

        Inputs:
        ----------------------------------
        trip_od: list of origin destination pairs
        scaled_trip_volume: scaled ground-truth of volume of trip which serves as our target.
        g: DGL graph object

        Outputs:
        ----------------------------------
        loss: value of loss function
        '''
        # calculate the in/out flow of nodes
        # scaled back trip volume
        # get in/out nodes of this batch
        out_nodes, _ = torch.unique(trip_od[:, 0], return_inverse=True)
        in_nodes, _ = torch.unique(trip_od[:, 1], return_inverse=True)
        # scale the in/out flows of the nodes in this batch
        scaled_out_flows = torch.sqrt(out_flows[out_nodes])
        scaled_in_flows = torch.sqrt(in_flows[in_nodes])
        # get embeddings of each node from GNN
        origin_embedding = self.origin_forward(self.graph)
        destination_embedding = self.destination_forward(self.graph)
        # get edge prediction
        edge_prediction = self.predict_edge(origin_embedding, destination_embedding, trip_od)
        # get in/out flow prediction
        in_flow_prediction = self.predict_inflow(destination_embedding, in_nodes)
        out_flow_prediction = self.predict_outflow(origin_embedding, out_nodes)
        # get edge prediction loss
        edge_predict_loss = self.MSE(edge_prediction, scaled_trip_volume)
        # get in/out flow prediction loss
        in_predict_loss = self.MSE(in_flow_prediction, scaled_in_flows)
        out_predict_loss = self.MSE(out_flow_prediction, scaled_out_flows)
        # get regularization loss
        reg_loss = 0.5 * (self.regularization_loss(origin_embedding) + self.regularization_loss(destination_embedding))
        # return the overall loss
        total_loss = self.hyperparams['multitask_weights'][0] * edge_predict_loss + \
            self.hyperparams['multitask_weights'][1] * in_predict_loss + \
            self.hyperparams['multitask_weights'][2] * out_predict_loss + \
            self.hyperparams['reg_param'] * reg_loss
        return total_loss
    

    def MSE(self, y_hat, y):
        '''
        Root mean square
        '''
        limit = 20000
        if y_hat.shape[0] < limit:
            return torch.mean((y_hat - y)**2)
        else:
            acc_sqe_sum = 0 # accumulative squred error sum
            for i in range(0, y_hat.shape[0], limit):
                acc_sqe_sum += torch.sum((y_hat[i: i + limit] - y[i: i + limit]) ** 2)
            return acc_sqe_sum / y_hat.shape[0]

    def predict_edge(self, origin_embedding, destination_embedding, test_cells):
        '''
        using node embeddings to make prediction on given trip OD.
        '''
        # construct edge feature
        src_emb = origin_embedding[test_cells[:,0]]
        dst_emb = destination_embedding[test_cells[:,1]]
        # get predictions
        # edge_feat = torch.cat((src_emb, dst_emb), dim=1)
        # self.edge_regressor(edge_feat)
        return self.edge_regressor(src_emb, dst_emb)

    def predict_inflow(self, embedding, in_nodes_idx):
        # make prediction
        return self.in_regressor(embedding[in_nodes_idx])

    def predict_outflow(self, embedding, out_nodes_idx):
        # make prediction
        return self.out_regressor(embedding[out_nodes_idx])

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))
    

    def generate_mini_batches(self, train_y):
        '''
        generator of mini-batch samples
        '''
        # negative sampling to get negative data
        neg_samples = self.negative_sampling(train_y)
        # binding together
        if neg_samples is not None:
            samples = torch.cat((train_y, neg_samples), dim=0)
        else:
            samples = train_y
        # shuffle
        samples = samples[torch.randperm(samples.shape[0])]
        # cut to mini-batches and wrap them by a generator
        for i in range(0, samples.shape[0], self.hyperparams['batch_size']):
            yield samples[i:i+self.hyperparams['batch_size']]

    def negative_sampling(self, train_y):
        '''
        perform negative sampling by perturbing the positive samples
        '''
        # if do not require negative sampling
        if self.hyperparams['negative_sampling_rate'] == 0:
            return None
        # else, let's do negative sampling
        # number of negative samples
        size_of_batch = len(train_y)
        num_to_generate = size_of_batch * self.hyperparams['negative_sampling_rate']
        # create container for negative samples
        neg_samples = np.tile(train_y, [self.hyperparams['negative_sampling_rate'], 1])
        neg_samples[:, -1] = 0 # set trip volume to be 0
        # perturbing the edge
        sample_nid = np.random.randint(self.num_regions, size = num_to_generate) # randomly sample nodes
        pos_choices = np.random.uniform(size = num_to_generate) # randomly sample position
        subj = pos_choices > 0.5
        obj = pos_choices <= 0.5
        neg_samples[subj, 0] = sample_nid[subj]
        neg_samples[obj, 1] = sample_nid[obj]
        # sanity check
        while(True):
            # check overlap edges
            overlap = npi.contains(train_y[:, :2], neg_samples[:, :2]) # True means overlap
            if overlap.any(): # if there is any overlap edge, resample for these edges
                # get the overlap subset
                neg_samples_overlap = neg_samples[overlap]
                # resample
                sample_nid = np.random.randint(self.num_regions, size = overlap.sum())
                pos_choices = np.random.uniform(size = overlap.sum())
                subj = pos_choices > 0.5
                obj = pos_choices <= 0.5
                neg_samples_overlap[subj, 0] = sample_nid[subj]
                neg_samples_overlap[obj, 1] = sample_nid[obj]
                # reassign the subset
                neg_samples[overlap] = neg_samples_overlap
            else: # if no overlap, just break resample loop
                break
        # return negative samples
        return torch.from_numpy(neg_samples)

    def predict(self, test_cells):
        '''
        predict based on trained model.
        '''
        # Evaluate model
        self.eval()
        with torch.no_grad():
            # get embedding
            origin_embedding = self.origin_forward(self.graph)
            destination_embedding = self.destination_forward(self.graph)
            # get prediction
            scaled_prediction = self.predict_edge(origin_embedding, destination_embedding, test_cells)
            # transform back to the original scale
            prediction = scaled_prediction ** 2
        return prediction.squeeze()

    def train_single(self, train_y, train_inflow, train_outflow):
        # turn model state
        self.train()
        
        # SGD from each mini-batch
        for mini_batch in tqdm(
            self.generate_mini_batches(train_y),
            total = len(range(0, train_y.shape[0], self.hyperparams['batch_size'])),
            disable = self.tqdm_disabled,
            desc = "SGD for each minibatch",
            leave = False
        ):
            # clear gradients
            self.optimizer.zero_grad()

            # get trip od
            trip_od = mini_batch[:, :2].long().to(self.device)
            # get trip volume
            scaled_trip_volume = torch.sqrt(mini_batch[:, -1].float()).to(self.device)
            # evaluate loss
            loss = self.get_loss(
                trip_od, 
                scaled_trip_volume, 
                train_inflow, 
                train_outflow
            )

            # back propagation to get gradients
            loss.backward()

            # clip to make stable
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.hyperparams['grad_norm'])

            # update weights by optimizer
            self.optimizer.step()
        
        # scheduler update learning rate
        self.scheduler.step()
    
    def run_single(self, train_y, train_inflow, train_outflow, test_index):
        
        # Train once
        self.train_single(train_y, train_inflow, train_outflow)
        
        # Test/predict
        return self.predict(test_index)

class GAT(nn.Module):

    def __init__(
            self, 
            graph, 
            num_regions, 
            input_size, 
            nodes_per_layer, 
            output_size, 
            num_hidden_layers, 
            dropout, 
            device='cpu'
        ):
        # initialize super class
        super().__init__()
        # handle the parameters
        self.graph = graph
        self.num_regions = num_regions
        self.input_size = input_size
        self.nodes_per_layer = nodes_per_layer
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.device = device
        # create gcn layers
        self.build_model()
    
    def build_model(self):
        self.layers = nn.ModuleList()
        # layer: input to hidden
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # layer: hidden to hidden
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # layer: hidden to output
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)
    
    def build_input_layer(self):
        act = F.relu
        return GATInputLayer(self.graph, self.input_size, self.nodes_per_layer)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return GATLayer(self.graph, self.nodes_per_layer, self.nodes_per_layer)
        
    def build_output_layer(self):
        return None

    def forward(self, graph):
        h = graph.ndata['attr']
        for i,layer in enumerate(self.layers):
            h = layer(h)
        return h


class Bilinear(nn.Module):

    def __init__(self, num_features, dropout=0, device='cpu'):
        return super().__init__()
        # bilinear
        self.bilinear = nn.Bilinear(num_features, num_features, 1)
        # dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, x1, x2):
        return self.bilinear(x1, x2)

class FNN(nn.Module):

    def __init__(self, num_features, dropout=0, device=False):
        # init super class
        super().__init__()
        # handle parameters
        self.in_feat = num_features
        self.h1_feat = num_features // 2
        self.h2_feat = self.h1_feat // 2
        self.out_feat = 1
        self.device = device
        # dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        # define functions
        self.linear1 = nn.Linear(self.in_feat, self.h1_feat)
        self.linear2 = nn.Linear(self.h1_feat, self.h2_feat)
        self.linear3 = nn.Linear(self.h2_feat, self.out_feat)
        self.activation = F.relu
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        # x = F.relu(x) # enforce the prediction to be non-zero
        return x


class GATLayer(nn.Module):

    def __init__(
            self, 
            graph, 
            in_ndim, 
            out_ndim,
            in_edim=1, 
            out_edim=1
        ):
        '''
        g: the graph
        input_size: input node feature dimension
        output_size: output node feature dimension
        edf_dim: input edge feature dimension
        '''
        super(GATLayer, self).__init__()
        self.graph = graph
        # equation (1)
        self.fc0 = nn.Linear(in_edim, out_edim, bias=False)
        self.fc1 = nn.Linear(in_ndim, out_ndim, bias=False)
        self.fc2 = nn.Linear(in_ndim, out_ndim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_ndim + out_edim, 1, bias=False)
        # equation (4)
        self.activation = F.relu
        # parameters
        self.weights = nn.Parameter(torch.Tensor(2, 1)) # used to calculate convex combination weights
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

    def edge_feat_func(self, edges):
        '''
        deal with edge features
        '''
        return {'t': self.fc0(edges.data['d'])}

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['t']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'],'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4). this is the core update part.
        z_neighbor = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        z_i = nodes.data['z_i']
        # calculate the convex combination weights
        lambda_ = F.softmax(self.weights, dim=0)
        # update
        h = self.activation(z_i + z_neighbor)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        self.graph.apply_edges(self.edge_feat_func)
        
        z = self.fc1(h) 
        self.graph.ndata['z'] = z
        z_i = self.fc2(h) 
        self.graph.ndata['z_i'] = z_i
        # equation (2)
        self.graph.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.graph.update_all(self.message_func, self.reduce_func)
        return self.graph.ndata.pop('h')
    
class GATInputLayer(nn.Module):

    def __init__(
            self, 
            graph, 
            in_ndim, 
            out_ndim, 
            in_edim=1, 
            out_edim=1
        ):
        '''
        g: the graph
        in_ndim: input node feature dimension
        out_ndim: output node feature dimension
        in_edim: input edge feature dimension
        out_edim: output edge feature dimension
        dropout: dropout rate
        '''
        # initialize super class
        super().__init__()
        # handle parameters
        self.graph = graph
        # equation (1)
        self.fc0 = nn.Linear(in_edim, out_edim, bias=False)
        self.fc1 = nn.Linear(in_ndim, out_ndim, bias=False)
        self.fc2 = nn.Linear(in_ndim, out_ndim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_ndim + out_edim, 1, bias=False)
        # equation (4)
        self.activation = F.relu
        # parameters
        self.weights = nn.Parameter(torch.Tensor(2, 1)) # used to calculate convex combination weights
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

    def edge_feat_func(self, edges):
        '''
        transform edge features
        '''
        return {'t': self.fc0(edges.data['d'])}

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['t']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4). this is the core update part.
        z_neighbor = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        z_i = nodes.data['z_i']
        # calculate the convex combination weights
        lambda_ = F.softmax(self.weights, dim=0)
        # update
        h = self.activation(z_i + z_neighbor)
        return {'h': h}

    def forward(self, attr):
        # equation (1)
        self.graph.apply_edges(self.edge_feat_func)
        z = self.fc1(attr) # message passed to the others
        self.graph.ndata['z'] = z
        z_i = self.fc2(attr) # message passed to self
        self.graph.ndata['z_i'] = z_i
        # equation (2)
        self.graph.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.graph.update_all(self.message_func, self.reduce_func)
        return self.graph.ndata.pop('h')