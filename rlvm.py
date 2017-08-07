"""Rectified Latent Variable Model for fitting neural time-series data

@author: Matt Whiteway, August 2017

TODO:

"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from FFnetwork.ffnetwork import FFNetwork
from FFnetwork.network import Network


class RLVM(Network):
    """Tensorflow (tf) implementation of RLVM class

    Attributes:
        network (FFNetwork object): autoencoder network
        input_size (int): size of input
        noise_dist (str): noise distribution used for cost function

        activations (list of tf ops): evaluates the layer-wise activations of
            the model
        cost (tf op): evaluates the cost function of the model
        cost_reg (tf op): evaluates the regularization penalty of the model
        cost_penalized (tf op): evaluates sum of `cost` and `cost_reg`
        train_step (tf op): evaluates one training step using the specified 
            cost function and learning algorithm

        data_in_ph (tf.placeholder): placeholder for input data
        data_out_ph (tf.placeholder): placeholder for output data
        data_in_var (tf.Variable): Variable for input data fed by data_in_ph
        data_out_var (tf.Variable): Variable for output data fed by data_out_ph
        data_in_batch (tf.Variable): batched version of data_in_var
        data_out_batch (tf.Variable): batched version of data_out_var
        indices (tf.placeholder): placeholder for indices into data_in/out_var
            to creat data_in/out_batch

        learning_alg (str): algorithm used for learning parameters
        learning_rate (float): learning rate used by the gradient 
            descent-based optimizers
        num_examples (int): total number of examples in training data
        use_batches (bool): all training data is placed on the GPU by 
            piping data into i/o variables through a feed_dict. If use_batches  
            is False, all data from these variables are used during training.
            If use_batches is True, the a separate index feed_dict is used to
            specify which chunks of data from these variables acts as input to
            the model.

        graph (tf.Graph): dataflow graph for the network
        use_gpu (bool): input into sess_config
        sess_config (tf.ConfigProto): specifies device for model training
        saver (tf.train.Saver): for saving and restoring variables
        merge_summaries (tf op): op that merges all summary ops
        init (tf op): op that initializes global variables in graph  

    """

    _allowed_noise_dists = ['gaussian', 'poisson', 'bernoulli']
    _allowed_learning_algs = ['adam', 'lbfgs']

    def __init__(
            self,
            layer_sizes=None,
            num_examples=None,
            act_funcs='relu',
            noise_dist='poisson',
            init_type='trunc_normal',
            learning_alg='adam',
            learning_rate=1e-3,
            use_batches=True,
            tf_seed=0,
            use_gpu=None):
        """Constructor for RLVM class

        Args:
            layer_sizes (list of ints): size of each layer, including input 
                layer
            num_examples (int): total number of examples in training data
            act_funcs (str or list of strs, optional): activation function for 
                network layers; replicated if a single element.
                ['relu'] | 'sigmoid' | 'tanh' | 'identity' | 'softplus' | 'elu'
            noise_dist (str, optional): noise distribution used by network
                ['gaussian'] | 'poisson' | 'bernoulli'
            init_type (str or list of strs, optional): initialization
                used for network weights; replicated if a single element.
                ['trunc_normal'] | 'normal' | 'zeros' | 'xavier'
            learning_alg (str, optional): algorithm used for learning 
                parameters. 
                ['adam'] | 'lbfgs'
            learning_rate (float, optional): learning rate used by the 
                gradient descent-based optimizers ('adam'). Default is 1e-3.
            use_batches (boolean, optional): determines how data is fed to 
                model; if False, all data is pinned to variables used 
                throughout fitting; if True, a data slicer and batcher are 
                constructed to feed the model shuffled batches throughout 
                training
            tf_seed (scalar, optional): rng seed for tensorflow to facilitate 
                building reproducible models
            use_gpu (bool): use gpu for training model

        Raises:
            TypeError: If layer_sizes argument is not specified
            TypeError: If num_examples argument is not specified
            ValueError: If noise_dist argument is not valid string
            ValueError: If learning_alg is not a valid string

        """

        # call __init__() method of super class
        super(RLVM, self).__init__()

        # input checking
        if layer_sizes is None:
            raise TypeError('Must specify layer sizes for network')
        if num_examples is None:
            raise TypeError('Must specify number of training examples')
        if noise_dist not in self._allowed_noise_dists:
            raise ValueError('Invalid noise distribution')
        if learning_alg not in self._allowed_learning_algs:
            raise ValueError('Invalid learning algorithm')

        # set model attributes from input
        self.input_size = layer_sizes[0]
        self.output_size = self.input_size
        self.noise_dist = noise_dist
        self.learning_alg = learning_alg
        self.learning_rate = learning_rate
        self.num_examples = num_examples
        self.use_batches = use_batches

        # for saving and restoring models
        self.graph = tf.Graph()  # must be initialized before graph creation

        # for specifying device
        if use_gpu is not None:
            self.use_gpu = use_gpu
            if use_gpu:
                self.sess_config = tf.ConfigProto(device_count={'GPU': 1})
            else:
                self.sess_config = tf.ConfigProto(device_count={'GPU': 0})

        # build model graph
        with self.graph.as_default():

            tf.set_random_seed(tf_seed)

            # define pipeline for feeding data into model
            with tf.name_scope('data'):
                self._initialize_data_pipeline()

            # initialize weights and create model
            self.network = FFNetwork(
                scope='model',
                inputs=self.data_in_batch,
                layer_sizes=layer_sizes,
                activation_funcs=act_funcs,
                weights_initializer=init_type,
                biases_initializer='zeros',
                log_activations=True)
            # set l2 penalty on weights to 0.0 to avoid future graph extensions
            for layer in range(self.network.num_layers):
                self.network.layers[layer].reg.vals['l2'] = 0.0

            # define loss function
            with tf.name_scope('loss'):
                self._define_loss()

            # define optimization routine
            with tf.name_scope('optimizer'):
                self._define_optimizer()

            # add additional ops
            # for saving and restoring models (initialized after var creation)
            self.saver = tf.train.Saver()
            # collect all summaries into a single op
            self.merge_summaries = tf.summary.merge_all()
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()

    def _define_loss(self):
        """Loss function that will be used to optimize model parameters"""

        data_out = self.data_out_batch
        pred = self.network.layers[-1].outputs

        # define cost function
        if self.noise_dist == 'gaussian':
            with tf.name_scope('gaussian_loss'):
                self.cost = tf.nn.l2_loss(data_out - pred)
        elif self.noise_dist == 'poisson':
            with tf.name_scope('poisson_loss'):
                cost = -tf.reduce_sum(
                    tf.multiply(data_out,
                                tf.log(self._log_min + pred))
                    - pred)
                # normalize by number of spikes
                self.cost = tf.divide(cost, tf.reduce_sum(data_out))
        elif self.noise_dist == 'bernoulli':
            with tf.name_scope('bernoulli_loss'):
                self.cost = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=data_out,
                        logits=pred))

        # add regularization penalties
        with tf.name_scope('regularization'):
            self.cost_reg = self.network.define_regularization_loss()

        self.cost_penalized = tf.add(self.cost, self.cost_reg)

        # save summary of cost
        with tf.name_scope('summaries'):
            tf.summary.scalar('cost', cost)

    def _assign_model_params(self, sess):
        """Functions assigns parameter values to randomly initialized model"""
        with self.graph.as_default():
            self.network.assign_model_params(sess)

    def _assign_reg_vals(self, sess):
        """Loops through all current regularization penalties and updates
        prameter values"""
        with self.graph.as_default():
            self.network.assign_reg_vals(sess)

    def set_regularization(self, reg_type, reg_val, layer_target=None):
        """Add or reassign regularization values

        Args:
            reg_type (str): see allowed_reg_types in regularization.py
            reg_val (int): corresponding regularization value
            layer_target (int or list of ints): specifies which layers the 
                current reg_type/reg_val pair is applied to

        """

        if layer_target is None:
            # set all layers
            layer_target = range(self.network.num_layers)

        # set regularization at the layer level
        rebuild_graph = False
        for layer in layer_target:
            new_reg_type = self.network.layers[layer].set_regularization(
                reg_type, reg_val)
            rebuild_graph = rebuild_graph or new_reg_type

        if rebuild_graph:
            with self.graph.as_default():
                # redefine loss function
                with tf.name_scope('loss'):
                    self._define_loss()

                # redefine optimization routine
                with tf.variable_scope('optimizer'):
                    self._define_optimizer()

    def forward_pass(self, input_data=None, output_data=None, data_indxs=None):
        """Transform a given input into its reconstruction 

        Args:
            input_data (time x input_dim numpy array): input to network
            output_data (time x output_dim numpy array): desired output of 
                network
            data_indxs (numpy array, optional): indexes of data to use in 
                calculating forward pass; if not supplied, all data is used

        Returns: 
            pred (time x num_cells numpy array): predicted model output

        """

        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self._restore_params(sess, input_data, output_data)

            # calculate model prediction
            pred = sess.run(
                self.network.layers[-1].outputs,
                feed_dict={self.indices: data_indxs})

        return pred

    def get_cost(self, input_data=None, output_data=None, data_indxs=None):
        """Get cost from loss function and regularization terms 

        Args:
            input_data (time x input_dim numpy array): input to model
            output_data (time x output_dim numpy array): desired output of 
                model
            data_indxs (numpy array, optional): indexes of data to use in 
                calculating forward pass; if not supplied, all data is used    

        Returns:
            cost (float): value of model's cost function evaluated on previous 
                model data or that used as input
            reg_pen (float): value of model's regularization penalty

        Raises:
            ValueError: If data_out is not supplied with data_in
            ValueError: If data_in/out time dims don't match

        """

        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self._restore_params(sess, input_data, output_data)

            cost, cost_reg = sess.run(
                [self.cost, self.cost_reg],
                feed_dict={self.indices: data_indxs})

        return cost, cost_reg

    def get_r2s(self, input_data=None, output_data=None, data_indxs=None):
        """Transform a given input into its reconstruction 

        Args:
            input_data (time x input_dim numpy array): input to network
            output_data (time x output_dim numpy array): desired output of 
                network
            data_indxs (numpy array, optional): indexes of data to use in 
                calculating forward pass; if not supplied, all data is used             

        Returns:
            r2s (1 x num_cells numpy array): pseudo-r2 values for each cell
            lls (dict): contains log-likelihoods for fitted model, null model 
                (prediction is mean), and saturated model (prediction is true
                activity) for each cell

        Raises:
            ValueError: If both input and output data are not provided
            ValueError: If input/output time dims don't match

        """

        if data_indxs is None:
            data_indxs = np.arange(self.num_examples)

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:

            self._restore_params(sess, input_data, output_data)

            data_in = sess.run(
                self.data_in_batch,
                feed_dict={self.indices: data_indxs})
            data_out = sess.run(
                self.data_out_batch,
                feed_dict={self.indices: data_indxs})
            pred = sess.run(
                self.network.layers[-1].outputs,
                feed_dict={self.indices: data_indxs})

        t, num_cells = data_in.shape
        mean_act = np.tile(np.mean(data_out, axis=0), (t, 1))

        if self.noise_dist == 'gaussian':

            ll = np.sum(np.square(data_out - pred), axis=0)
            ll_null = np.sum(np.square(data_out - mean_act), axis=0)
            ll_sat = 0.0

        elif self.noise_dist == 'poisson':

            ll = -np.sum(
                np.multiply(data_out, np.log(self._log_min + pred))
                - pred, axis=0)
            ll_null = -np.sum(
                np.multiply(data_out, np.log(self._log_min + mean_act))
                - mean_act, axis=0)
            ll_sat = np.multiply(data_out, np.log(self._log_min + data_out))
            ll_sat = -np.sum(ll_sat - data_out, axis=0)

        elif self.noise_dist == 'bernoulli':

            ll_sat = 1.0
            ll_null = 0.0
            ll = 0.0

        r2s = 1.0 - np.divide(ll_sat - ll, ll_sat - ll_null)
        r2s[ll_sat == ll_null] = 1.0

        lls = {
            'll': ll,
            'll_null': ll_null,
            'll_sat': ll_sat
        }

        return r2s, lls

    def get_reg_pen(self):
        """Return reg penalties in a dictionary"""

        reg_dict = {}
        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            # initialize all parameters randomly
            sess.run(self.init)

            # overwrite randomly initialized values of model with stored values
            self._assign_model_params(sess)

            # update regularization parameter values
            self._assign_reg_vals(sess)

            with tf.name_scope('get_reg_pen'):  # to keep the graph clean-ish
                for layer in range(self.network.num_layers):
                    reg_dict['layer%i' % layer] = \
                        self.network.layers[layer].get_reg_pen(sess)

        return reg_dict
