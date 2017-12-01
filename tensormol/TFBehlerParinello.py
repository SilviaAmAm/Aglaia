"""
Behler-Parinello network classes
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random

from TensorMol.TensorMolData import *
from TensorMol.RawEmbeddings import *
from tensorflow.python.client import timeline

class BehlerParinelloDirectSymFunc:
    """
    Behler-Parinello network using embedding from RawEmbeddings.py
    """
    def __init__(self, molecule_set=None, embedding_type=None, name=None):
        """
        Args:
            tensor_data (TensorMol.TensorMolData object): a class which holds the training data
            name (str): a name used to recall this network

        Notes:
            if name != None, attempts to load a previously saved network, otherwise assumes a new network
        """
        #Network and training parameters
        self.tf_precision = eval(PARAMS["tf_prec"])
        TensorMol.RawEmbeddings.data_precision = self.tf_precision
        self.hidden_layers = PARAMS["HiddenLayers"]
        self.learning_rate = PARAMS["learning_rate"]
        self.weight_decay = PARAMS["weight_decay"]
        self.momentum = PARAMS["momentum"]
        self.max_steps = PARAMS["max_steps"]
        self.batch_size = PARAMS["batch_size"]
        self.max_checkpoints = PARAMS["max_checkpoints"]
        self.train_energy_gradients = PARAMS["train_energy_gradients"]
        self.profiling = PARAMS["Profiling"]
        self.activation_function_type = PARAMS["NeuronType"]
        self.randomize_data = PARAMS["RandomizeData"]
        self.test_ratio = PARAMS["TestRatio"]
        self.assign_activation()
        self.embedding_type = embedding_type
        self.path = PARAMS["networks_directory"]

        #Reloads a previous network if name variable is not None
        if name !=  None:
            self.name = name
            self.load_network()
            LOGGER.info("Reloaded network from %s", self.network_directory)
            return

        #Data parameters
        self.molecule_set = molecule_set
        self.molecule_set_name = self.molecule_set.name
        self.elements = self.molecule_set.AtomTypes()
        self.max_num_atoms = self.molecule_set.MaxNAtoms()
        self.num_molecules = len(self.molecule_set.mols)
        self.network_type = "BehlerParinelloDirectSymFunc"
        self.name = self.network_type+"_"+self.molecule_set_name+"_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")
        self.network_directory = './networks/'+self.name

        self.set_symmetry_function_params()

        LOGGER.info("self.learning_rate: %d", self.learning_rate)
        LOGGER.info("self.batch_size: %d", self.batch_size)
        LOGGER.info("self.max_steps: %d", self.max_steps)
        return

    def sigmoid_with_param(self, x):
        return tf.log(1.0+tf.exp(tf.multiply(tf.cast(PARAMS["sigmoid_alpha"], dtype=self.tf_precision), x)))/tf.cast(PARAMS["sigmoid_alpha"], dtype=self.tf_precision)

    def assign_activation(self):
        LOGGER.debug("Assigning Activation Function: %s", PARAMS["NeuronType"])
        try:
            if self.activation_function_type == "relu":
                self.activation_function = tf.nn.relu
            elif self.activation_function_type == "elu":
                self.activation_function = tf.nn.elu
            elif self.activation_function_type == "selu":
                self.activation_function = self.selu
            elif self.activation_function_type == "softplus":
                self.activation_function = tf.nn.softplus
            elif self.activation_function_type == "tanh":
                self.activation_function = tf.tanh
            elif self.activation_function_type == "sigmoid":
                self.activation_function = tf.sigmoid
            elif self.activation_function_type == "sigmoid_with_param":
                self.activation_function = self.sigmoid_with_param
            else:
                print ("unknown activation function, set to relu")
                self.activation_function = tf.nn.relu
        except Exception as Ex:
            print(Ex)
            print ("activation function not assigned, set to relu")
            self.activation_function = tf.nn.relu
        return

    def save_checkpoint(self, step):
        checkpoint_file = os.path.join(self.network_directory,self.name+'-checkpoint-'+str(step))
        LOGGER.info("Saving checkpoint file %s", checkpoint_file)
        self.saver.save(self.sess, checkpoint_file)
        return

    def save_network(self):
        print("Saving TFInstance")
        self.clean_scratch()
        self.clean()
        f = open(self.network_directory+".tfn","wb")
        pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        return

    def load_network(self):
        LOGGER.info("Loading TFInstance")
        f = open(self.path+"/"+self.name+".tfn","rb")
        import TensorMol.PickleTM
        network_member_variables = TensorMol.PickleTM.UnPickleTM(f)
        self.clean()
        self.__dict__.update(network_member_variables)
        f.close()
        checkpoint_files = [x for x in os.listdir(self.network_directory) if (x.count('checkpoint')>0 and x.count('meta')==0)]
        if (len(checkpoint_files)>0):
            self.latest_checkpoint_file = checkpoint_files[0]
        else:
            LOGGER.error("Network not found in directory: %s", self.network_directory)
            LOGGER.error("Network directory contents: %s", str(os.listdir(self.network_directory)))
        return

    def clean_scratch(self):
        self.scratch_state = None
        self.scratch_pointer = 0 # for non random batch iteration.
        self.scratch_train_inputs = None
        self.scratch_train_outputs = None
        self.scratch_test_inputs = None # These should be partitioned out by LoadElementToScratch
        self.scratch_test_outputs = None
        self.molecule_set = None
        self.xyz_data = None
        self.Z_data = None
        self.energy_data = None
        self.gradient_data = None
        self.num_atoms_data = None
        return

    def reload_set(self):
        """
        Recalls the MSet to build training data etc.
        """
        self.molecule_set = MSet(self.molecule_set_name)
        self.molecule_set.Load()
        return

    def load_data(self):
        if (self.molecule_set == None):
            try:
                self.reload_set()
            except Exception as Ex:
                print("TensorData object has no molecule set.", Ex)
        if self.randomize_data:
            random.shuffle(self.molecule_set.mols)
        xyzs = np.zeros((self.num_molecules, self.max_num_atoms, 3), dtype = np.float32)
        Zs = np.zeros((self.num_molecules, self.max_num_atoms), dtype = np.int32)
        num_atoms = np.zeros((self.num_molecules), dtype = np.int32)
        energies = np.zeros((self.num_molecules), dtype = np.float32)
        gradients = np.zeros((self.num_molecules, self.max_num_atoms, 3), dtype=np.float32)
        for i, mol in enumerate(self.molecule_set.mols):
            xyzs[i][:mol.NAtoms()] = mol.coords
            Zs[i][:mol.NAtoms()] = mol.atoms
            energies[i] = mol.properties["atomization"]
            num_atoms[i] = mol.NAtoms()
            gradients[i][:mol.NAtoms()] = mol.properties["gradients"]
        return xyzs, Zs, energies, num_atoms, gradients

    def load_data_to_scratch(self):
        """
        Reads built training data off disk into scratch space.
        Divides training and test data.
        Normalizes inputs and outputs.
        note that modifies my MolDigester to incorporate the normalization
        Initializes pointers used to provide training batches.

        Args:
            random: Not yet implemented randomization of the read data.

        Note:
            Also determines mean stoichiometry
        """
        self.xyz_data, self.Z_data, self.energy_data, self.num_atoms_data, self.gradient_data = self.load_data()
        self.num_test_cases = int(self.test_ratio * self.num_molecules)
        self.last_train_case = int(self.num_molecules - self.num_test_cases)
        self.num_train_cases = self.last_train_case
        self.test_scratch_pointer = self.last_train_case
        self.train_scratch_pointer = 0
        LOGGER.debug("Number of training cases: %i", self.num_train_cases)
        LOGGER.debug("Number of test cases: %i", self.num_test_cases)
        return

    def get_energy_train_batch(self, batch_size):
        if batch_size > self.num_train_cases:
            raise Exception("Insufficent training data to fill a training batch.\n"\
                    +str(self.num_train_cases)+" cases in dataset with a batch size of "+str(batch_size))
        if self.train_scratch_pointer + batch_size >= self.num_train_cases:
            self.train_scratch_pointer = 0
        self.train_scratch_pointer += batch_size
        xyzs = self.xyz_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        Zs = self.Z_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        energies = self.energy_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        num_atoms = self.num_atoms_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        gradients = self.gradient_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        return [xyzs, Zs, energies, gradients, num_atoms]

    def get_energy_test_batch(self, batch_size):
        if batch_size > self.num_test_cases:
            raise Exception("Insufficent training data to fill a test batch.\n"\
                    +str(self.num_test_cases)+" cases in dataset with a batch size of "+str(num_cases_batch))
        if self.test_scratch_pointer + batch_size >= self.num_train_cases:
            self.test_scratch_pointer = self.last_train_case
        self.test_scratch_pointer += batch_size
        xyzs = self.xyz_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        Zs = self.Z_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        energies = self.energy_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        num_atoms = self.num_atoms_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        gradients = self.gradient_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        return [xyzs, Zs, energies, gradients, num_atoms]

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def variable_with_weight_decay(self, shape, stddev, weight_decay, name = None):
        """
        Creates random tensorflow variable from a truncated normal distribution with weight decay

        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

        Returns:
            Variable Tensor

        Notes:
            Note that the Variable is initialized with a truncated normal distribution.
            A weight decay is added only if one is specified.
        """
        variable = tf.Variable(tf.truncated_normal(shape, stddev = stddev, dtype = self.tf_precision), name = name)
        if weight_decay is not None:
            weightdecay = tf.multiply(tf.nn.l2_loss(variable), weight_decay, name='weight_loss')
            tf.add_to_collection('losses', weightdecay)
        return variable

    def compute_normalization(self):
        elements = tf.constant(self.elements, dtype = tf.int32)
        element_pairs = tf.constant(self.element_pairs, dtype = tf.int32)
        radial_rs = tf.constant(self.radial_rs, dtype = self.tf_precision)
        angular_rs = tf.constant(self.angular_rs, dtype = self.tf_precision)
        theta_s = tf.constant(self.theta_s, dtype = self.tf_precision)
        radial_cutoff = tf.constant(self.radial_cutoff, dtype = self.tf_precision)
        angular_cutoff = tf.constant(self.angular_cutoff, dtype = self.tf_precision)
        zeta = tf.constant(self.zeta, dtype = self.tf_precision)
        eta = tf.constant(self.eta, dtype = self.tf_precision)
        xyzs_pl = tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms, 3]))
        Zs_pl = tf.placeholder(tf.int32, shape=tuple([self.batch_size, self.max_num_atoms]))
        embeddings, molecule_indices = tf_symmetry_functions(xyzs_pl, Zs_pl, elements, element_pairs, radial_cutoff,
                                        angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)
        embeddings_list = [[], [], [], []]
        labels_list = []
        gradients_list = []

        self.embeddings_max = []
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for ministep in range (0, max(2, int(0.1 * self.num_train_cases/self.batch_size))):
            batch_data = self.get_energy_train_batch(self.batch_size)
            num_atoms = batch_data[4]
            labels_list.append(batch_data[2])
            for molecule in range(self.batch_size):
                gradients_list.append(batch_data[3][molecule,:num_atoms[molecule]])
            embedding, molecule_index = sess.run([embeddings, molecule_indices], feed_dict = {xyzs_pl:batch_data[0], Zs_pl:batch_data[1]})
            for element in range(len(self.elements)):
                embeddings_list[element].append(embedding[element])
        sess.close()
        for element in range(len(self.elements)):
            self.embeddings_max.append(np.amax(np.concatenate(embeddings_list[element])))
        labels = np.concatenate(labels_list)
        self.labels_mean = np.mean(labels)
        self.labels_stddev = np.std(labels)
        gradients = np.concatenate(gradients_list)
        self.gradients_mean = np.mean(gradients)
        self.gradients_stddev = np.std(gradients)
        self.train_scratch_pointer = 0

        #Set the embedding and label shape
        self.embedding_shape = embedding[0].shape[1]
        self.label_shape = labels[0].shape
        return

    def set_symmetry_function_params(self):
        self.element_pairs = np.array([[self.elements[i], self.elements[j]] for i in range(len(self.elements)) for j in range(i, len(self.elements))])
        self.zeta = PARAMS["AN1_zeta"]
        self.eta = PARAMS["AN1_eta"]

        #Define radial grid parameters
        num_radial_rs = PARAMS["AN1_num_r_Rs"]
        self.radial_cutoff = PARAMS["AN1_r_Rc"]
        self.radial_rs = self.radial_cutoff * np.linspace(0, (num_radial_rs - 1.0) / num_radial_rs, num_radial_rs)

        #Define angular grid parameters
        num_angular_rs = PARAMS["AN1_num_a_Rs"]
        num_angular_theta_s = PARAMS["AN1_num_a_As"]
        self.angular_cutoff = PARAMS["AN1_a_Rc"]
        self.theta_s = 2.0 * np.pi * np.linspace(0, (num_angular_theta_s - 1.0) / num_angular_theta_s, num_angular_theta_s)
        self.angular_rs = self.angular_cutoff * np.linspace(0, (num_angular_rs - 1.0) / num_angular_rs, num_angular_rs)
        return

    def clean(self):
        self.sess = None
        self.total_loss = None
        self.train_op = None
        self.saver = None
        self.summary_writer = None
        self.summary_op = None
        self.activation_function = None
        self.options = None
        self.run_metadata = None
        self.xyzs_pl = None
        self.Zs_pl = None
        self.labels_pl = None
        self.num_atoms_pl = None
        self.gradients_pl = None
        self.energy_loss = None
        self.gradients_loss = None
        self.output = None
        self.gradients = None
        self.gradient_labels = None
        return

    def train_prepare(self,  continue_training =False):
        """
        Get placeholders, graph and losses in order to begin training.
        Also assigns the desired padding.

        Args:
            continue_training: should read the graph variables from a saved checkpoint.
        """
        with tf.Graph().as_default():
            #Define the placeholders to be fed in for each batch
            self.xyzs_pl = tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms, 3]))
            self.Zs_pl = tf.placeholder(tf.int32, shape=tuple([self.batch_size, self.max_num_atoms]))
            self.labels_pl = tf.placeholder(self.tf_precision, shape=tuple([self.batch_size]))
            self.gradients_pl = tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms, 3]))
            self.num_atoms_pl = tf.placeholder(tf.int32, shape=([self.batch_size]))

            #Define the constants/Variables for the symmetry function basis
            elements = tf.constant(self.elements, dtype = tf.int32)
            element_pairs = tf.constant(self.element_pairs, dtype = tf.int32)
            radial_rs = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
            angular_rs = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
            theta_s = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
            radial_cutoff = tf.constant(self.radial_cutoff, dtype = self.tf_precision)
            angular_cutoff = tf.constant(self.angular_cutoff, dtype = self.tf_precision)
            zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
            eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)

            #Define normalization constants
            embeddings_max = tf.constant(self.embeddings_max, dtype = self.tf_precision)
            labels_mean = tf.constant(self.labels_mean, dtype = self.tf_precision)
            labels_stddev = tf.constant(self.labels_stddev, dtype = self.tf_precision)
            gradients_mean = tf.constant(self.gradients_mean, dtype = self.tf_precision)
            gradients_stddev = tf.constant(self.gradients_stddev, dtype = self.tf_precision)
            num_atoms_batch = tf.reduce_sum(self.num_atoms_pl)

            #Define the graph for computing the embedding, feeding through the network, and evaluating the loss
            element_embeddings, mol_indices = tf_symmetry_functions(self.xyzs_pl, self.Zs_pl, elements,
                    element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)
            for element in range(len(self.elements)):
                element_embeddings[element] /= embeddings_max[element]
            normalized_output = self.inference(element_embeddings, mol_indices)
            self.output = (normalized_output * self.labels_stddev) + self.labels_mean

            self.gradients = tf.gather_nd(tf.gradients(self.output, self.xyzs_pl)[0], tf.where(tf.not_equal(self.Zs_pl, 0)))
            self.gradient_labels = tf.gather_nd(self.gradients_pl, tf.where(tf.not_equal(self.Zs_pl, 0)))
            if self.train_energy_gradients:
                self.total_loss, self.energy_loss, self.gradient_loss = self.loss_op(self.output,
                        self.labels_pl, self.gradients, self.gradient_labels, num_atoms_batch)
            else:
                self.total_loss, self.energy_loss = self.loss_op(self.output, self.labels_pl)
            self.train_op = self.optimizer(self.total_loss, self.learning_rate, self.momentum)
            self.summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
            self.summary_writer = tf.summary.FileWriter(self.network_directory, self.sess.graph)
            self.sess.run(init)
        return

    def fill_feed_dict(self, batch_data):
        """
        Fill the tensorflow feed dictionary.

        Args:
            batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
            and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

        Returns:
            Filled feed dictionary.
        """
        if (not np.all(np.isfinite(batch_data[2]),axis=(0))):
            print("I was fed shit")
            raise Exception("DontEatShit")
        feed_dict={i: d for i, d in zip([self.xyzs_pl, self.Zs_pl, self.labels_pl, self.gradients_pl, self.num_atoms_pl], batch_data)}
        return feed_dict

    def inference(self, inp, indexs):
        """
        Builds a Behler-Parinello graph

        Args:
            inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
            index: a list of (num_of atom type X batchsize) array which linearly combines the elements
        Returns:
            The BP graph output
        """
        branches=[]
        output = tf.zeros([self.batch_size, self.max_num_atoms], dtype=self.tf_precision)
        for e in range(len(self.elements)):
            branches.append([])
            inputs = inp[e]
            index = indexs[e]
            for i in range(len(self.hidden_layers)):
                if i == 0:
                    with tf.name_scope(str(self.elements[e])+'_hidden1'):
                        weights = self.variable_with_weight_decay(shape=[self.embedding_shape, self.hidden_layers[i]],
                                stddev=math.sqrt(2.0 / float(self.embedding_shape)), weight_decay=self.weight_decay, name="weights")
                        biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
                        branches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
                else:
                    with tf.name_scope(str(self.elements[e])+'_hidden'+str(i+1)):
                        weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
                                stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
                        biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
                        branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
            with tf.name_scope(str(self.elements[e])+'_regression_linear'):
                weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], 1],
                        stddev=math.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
                biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
                branches[-1].append(tf.squeeze(tf.matmul(branches[-1][-1], weights) + biases))
                output += tf.scatter_nd(index, branches[-1][-1], [self.batch_size, self.max_num_atoms])
            tf.verify_tensor_all_finite(output,"Nan in output!!!")
        return tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size])

    def optimizer(self, loss, learning_rate, momentum):
        """
        Sets up the training Ops.
        Creates a summarizer to track the loss over time in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
            loss: Loss tensor, from loss().
            learning_rate: the learning rate to use for gradient descent.

        Returns:
            train_op: the tensorflow operation to call for training.
        """
        tf.summary.scalar(loss.op.name, loss)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def loss_op(self, output, labels, gradients = None, gradient_labels = None, num_atoms = None):
        energy_loss = tf.nn.l2_loss(tf.subtract(output, labels))
        tf.add_to_collection('losses', energy_loss)
        if self.train_energy_gradients:
            gradients_loss = self.batch_size * tf.nn.l2_loss(tf.subtract(gradients, gradient_labels)) / tf.cast(num_atoms, self.tf_precision)
            tf.add_to_collection('losses', gradients_loss)
            return tf.add_n(tf.get_collection('losses'), name='total_loss'), energy_loss, gradients_loss
        else:
            return tf.add_n(tf.get_collection('losses'), name='total_loss'), energy_loss

    def train_step(self, step):
        """
        Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

        Args:
            step: the index of this step.
        """
        Ncase_train = self.num_train_cases
        start_time = time.time()
        train_loss =  0.0
        train_energy_loss = 0.0
        train_gradient_loss = 0.0
        num_mols = 0
        for ministep in range (0, int(Ncase_train/self.batch_size)):
            batch_data = self.get_energy_train_batch(self.batch_size)
            if self.train_energy_gradients:
                _, total_loss_value, energy_loss, gradient_loss, mol_output, n_atoms, gradients, gradient_labels = self.sess.run([self.train_op,
                        self.total_loss, self.energy_loss, self.gradient_loss, self.output, self.num_atoms_pl, self.gradients,
                        self.gradients_pl], feed_dict=self.fill_feed_dict(batch_data))
                train_gradient_loss += gradient_loss
            else:
                _, total_loss_value, energy_loss, mol_output = self.sess.run([self.train_op, self.total_loss,
                            self.energy_loss, self.output], feed_dict=self.fill_feed_dict(batch_data))
            train_loss += total_loss_value
            train_energy_loss += energy_loss
            num_mols += self.batch_size
        duration = time.time() - start_time
        if self.train_energy_gradients:
            self.print_training(step, train_loss, train_energy_loss, num_mols, duration, train_gradient_loss)
        else:
            self.print_training(step, train_loss, train_energy_loss, num_mols, duration)
        return

    def test_step(self, step):
        """
        Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

        Args:
            step: the index of this step.
        """
        print( "testing...")
        test_loss =  0.0
        start_time = time.time()
        Ncase_test = self.num_test_cases
        num_mols = 0
        test_energy_loss = 0.0
        test_gradient_loss = 0.0
        test_epoch_energy_labels, test_epoch_energy_outputs = [], []
        test_epoch_force_labels, test_epoch_force_outputs = [], []
        num_atoms_epoch = 0.0
        for ministep in range (0, int(Ncase_test/self.batch_size)):
            batch_data = self.get_energy_test_batch(self.batch_size)
            feed_dict = self.fill_feed_dict(batch_data)
            if self.train_energy_gradients:
                output, labels, gradients, gradient_labels, total_loss_value, energy_loss, gradient_loss, num_atoms_batch = self.sess.run([self.output,
                        self.labels_pl, self.gradients, self.gradients_pl, self.total_loss, self.energy_loss, self.gradient_loss,
                        self.num_atoms_pl],  feed_dict=feed_dict)
                test_gradient_loss += gradient_loss
            else:
                output, labels, gradients, gradient_labels, total_loss_value, energy_loss, num_atoms_batch = self.sess.run([self.output,
                        self.labels_pl, self.gradients, self.gradient_labels, self.total_loss, self.energy_loss,
                        self.num_atoms_pl],  feed_dict=feed_dict)
            test_loss += total_loss_value
            num_mols += self.batch_size
            test_energy_loss += energy_loss
            test_epoch_energy_labels.append(labels)
            test_epoch_energy_outputs.append(output)
            num_atoms_epoch += np.sum(num_atoms_batch)
            test_epoch_force_labels.append(-1.0 * gradient_labels)
            test_epoch_force_outputs.append(-1.0 * gradients)
        test_epoch_energy_labels = np.concatenate(test_epoch_energy_labels)
        test_epoch_energy_outputs = np.concatenate(test_epoch_energy_outputs)
        test_epoch_energy_errors = test_epoch_energy_labels - test_epoch_energy_outputs
        test_epoch_force_labels = np.concatenate(test_epoch_force_labels)
        test_epoch_force_outputs = np.concatenate(test_epoch_force_outputs)
        test_epoch_force_errors = test_epoch_force_labels - test_epoch_force_outputs
        duration = time.time() - start_time
        for i in [random.randint(0, self.batch_size - 1) for _ in xrange(20)]:
            LOGGER.info("Energy label: %.8f  Energy output: %.8f", test_epoch_energy_labels[i], test_epoch_energy_outputs[i])
        for i in [random.randint(0, num_atoms_epoch - 1) for _ in xrange(20)]:
            LOGGER.info("Forces label: %s  Forces output: %s", test_epoch_force_labels[i], test_epoch_force_outputs[i])
        LOGGER.info("MAE  Energy: %11.8f    Forces: %11.8f", np.mean(np.abs(test_epoch_energy_errors)), np.mean(np.abs(test_epoch_force_errors)))
        LOGGER.info("MSE  Energy: %11.8f    Forces: %11.8f", np.mean(test_epoch_energy_errors), np.mean(test_epoch_force_errors))
        LOGGER.info("RMSE Energy: %11.8f    Forces: %11.8f", np.sqrt(np.mean(np.square(test_epoch_energy_errors))), np.sqrt(np.mean(np.square(test_epoch_force_errors))))
        if self.train_energy_gradients:
            self.print_testing(step, test_loss, test_energy_loss, num_mols, duration, test_gradient_loss)
        else:
            self.print_testing(step, test_loss, test_energy_loss, num_mols, duration)
        return test_loss

    def train(self):
        self.load_data_to_scratch()
        self.compute_normalization()
        self.train_prepare()
        test_freq = PARAMS["test_freq"]
        mini_test_loss = 100000000 # some big numbers
        for step in range(1, self.max_steps+1):
            self.train_step(step)
            if step%test_freq==0 and step!=0 :
                test_loss = self.test_step(step)
                if (test_loss < mini_test_loss):
                    mini_test_loss = test_loss
                    self.save_checkpoint(step)
        self.save_network()
        return

    def print_training(self, step, loss, energy_loss, num_mols, duration, gradient_loss=None):
        if self.train_energy_gradients:
            LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f  energy loss: %.10f  gradient loss: %.10f",
                        step, duration, loss / num_mols, energy_loss / num_mols, gradient_loss / num_mols)
        else:
            LOGGER.info("step: %7d  duration: %.5f  train loss: %.10f energy loss: %.10f",
                        step, duration, loss / num_mols, energy_loss / num_mols)
        return

    def print_testing(self, step, loss, energy_loss, num_mols, duration, gradient_loss=None):
        if self.train_energy_gradients:
            LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f  energy loss: %.10f  gradient loss: %.10f",
                        step, duration, loss / num_mols, energy_loss / num_mols, gradient_loss / num_mols)
        else:
            LOGGER.info("step: %7d  duration: %.5f  test loss: %.10f energy loss: %.10f",
                        step, duration, loss / num_mols, energy_loss / num_mols)
        return

    def evaluate_prepare(self):
        """
        Get placeholders, graph and losses in order to begin training.
        Also assigns the desired padding.

        Args:
            continue_training: should read the graph variables from a saved checkpoint.
        """
        with tf.Graph().as_default():
            #Define the placeholders to be fed in for each batch
            self.xyzs_pl = tf.placeholder(self.tf_precision, shape=tuple([self.batch_size, self.max_num_atoms, 3]))
            self.Zs_pl = tf.placeholder(tf.int32, shape=tuple([self.batch_size, self.max_num_atoms]))

            #Define the constants/Variables for the symmetry function basis
            elements = tf.constant(self.elements, dtype = tf.int32)
            element_pairs = tf.constant(self.element_pairs, dtype = tf.int32)
            radial_rs = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
            angular_rs = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
            theta_s = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
            radial_cutoff = tf.constant(self.radial_cutoff, dtype = self.tf_precision)
            angular_cutoff = tf.constant(self.angular_cutoff, dtype = self.tf_precision)
            zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
            eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)

            #Define normalization constants
            embeddings_max = tf.constant(self.embeddings_max, dtype = self.tf_precision)
            labels_mean = tf.constant(self.labels_mean, dtype = self.tf_precision)
            labels_stddev = tf.constant(self.labels_stddev, dtype = self.tf_precision)

            #Define the graph for computing the embedding, feeding through the network, and evaluating the loss
            element_embeddings, mol_indices = tf_symmetry_functions(self.xyzs_pl, self.Zs_pl, elements,
                    element_pairs, radial_cutoff, angular_cutoff, radial_rs, angular_rs, theta_s, zeta, eta)
            for element in range(len(self.elements)):
                element_embeddings[element] /= embeddings_max[element]
            self.normalized_output = self.inference(element_embeddings, mol_indices)
            self.output = (self.normalized_output * self.labels_stddev) + self.labels_mean
            self.gradients = tf.gradients(self.output, self.xyzs_pl)[0] / self.batch_size

            self.summary_op = tf.summary.merge_all()
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
            self.summary_writer = tf.summary.FileWriter(self.network_directory, self.sess.graph)
        return

    def evaluate_fill_feed_dict(self, xyzs, Zs):
        """
        Fill the tensorflow feed dictionary.

        Args:
            batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
            and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

        Returns:
            Filled feed dictionary.
        """
        feed_dict={i: d for i, d in zip([self.xyzs_pl, self.Zs_pl], [xyzs, Zs])}
        return feed_dict

    def evaluate_mol(self, mol, eval_forces=True):
        """
        Takes coordinates and atomic numbers from a manager and feeds them into the network
        for evaluation of the forces

        Args:
            xyzs (np.float): numpy array of atomic coordinates
            Zs (np.int32): numpy array of atomic numbers
        """
        if not self.sess:
            print(self.latest_checkpoint_file)
            print("loading the session..")
            self.batch_size = 1
            self.num_atoms = mol.NAtoms()
            self.assign_activation()
            self.evaluate_prepare()
        xyzs = np.expand_dims(mol.coords, axis=0)
        Zs = np.expand_dims(mol.atoms, axis=0)
        feed_dict=self.evaluate_fill_feed_dict(xyzs, Zs)
        energy = self.sess.run(self.output, feed_dict=feed_dict)
        return energy

    def evaluate_batch(self, mols, eval_forces=True):
        """
        Takes coordinates and atomic numbers from a manager and feeds them into the network
        for evaluation of the forces

        Args:
            xyzs (np.float): numpy array of atomic coordinates
            Zs (np.int32): numpy array of atomic numbers
        """
        if not self.sess:
            print(self.latest_checkpoint_file)
            print("loading the session..")
            self.batch_size = len(mols)
            self.max_num_atoms = max([mol.NAtoms() for mol in mols])
            self.assign_activation()
            self.evaluate_prepare()
        # if (max([mol.NAtoms() for mol in mols]) > self.max_num_atoms) or (len(mols) > self.batch_size):
     #         self.max_num_atoms = max([mol.NAtoms() for mol in mols])
        #     self.batch_size = len(mols)
        #     self.evaluate_prepare()
        xyzs = np.zeros((self.batch_size, self.max_num_atoms, 3))
        Zs = np.zeros((self.batch_size, self.max_num_atoms))
        for i, mol in enumerate(mols):
            xyzs[i, :mol.NAtoms()] = mol.coords
            Zs[i, :mol.NAtoms()] = mol.atoms
        feed_dict=self.evaluate_fill_feed_dict(xyzs, Zs)
        energy = self.sess.run(self.output, feed_dict=feed_dict)
        return energy[:len(mols)]

class BehlerParinelloDirectGauSH:
    """
    Behler-Parinello network using embedding from RawEmbeddings.py
    """
    def __init__(self, molecule_set=None, embedding_type=None, name=None):
        """
        Args:
            tensor_data (TensorMol.TensorMolData object): a class which holds the training data
            name (str): a name used to recall this network

        Notes:
            if name != None, attempts to load a previously saved network, otherwise assumes a new network
        """
        self.tf_precision = eval(PARAMS["tf_prec"])
        TensorMol.RawEmbeddings.data_precision = self.tf_precision
        self.hidden_layers = PARAMS["HiddenLayers"]
        self.learning_rate = PARAMS["learning_rate"]
        self.weight_decay = PARAMS["weight_decay"]
        self.momentum = PARAMS["momentum"]
        self.max_steps = PARAMS["max_steps"]
        self.batch_size = PARAMS["batch_size"]
        self.max_checkpoints = PARAMS["max_checkpoints"]
        self.train_energy_gradients = PARAMS["train_energy_gradients"]
        self.profiling = PARAMS["Profiling"]
        self.activation_function_type = PARAMS["NeuronType"]
        self.number_radial = PARAMS["SH_NRAD"]
        self.l_max = PARAMS["SH_LMAX"]
        self.gaussian_params = PARAMS["RBFS"][:self.number_radial]
        self.atomic_embed_factors = PARAMS["ANES"]
        self.randomize_data = PARAMS["RandomizeData"]
        self.test_ratio = PARAMS["TestRatio"]
        self.assign_activation()
        self.embedding_type = embedding_type
        self.path = PARAMS["networks_directory"]
        self.Ree_on = PARAMS["EECutoffOn"]
        self.Ree_off = PARAMS["EECutoffOff"]
        self.Ree_cut = PARAMS["EECutoffOff"]
        self.damp_shifted_alpha = PARAMS["DSFAlpha"]
        self.elu_width = PARAMS["Elu_Width"]
        self.elu_shift = DSF(self.elu_width*BOHRPERA, self.Ree_off*BOHRPERA, self.damp_shifted_alpha/BOHRPERA)
        self.elu_alpha = DSF_Gradient(self.elu_width*BOHRPERA, self.Ree_off*BOHRPERA, self.damp_shifted_alpha/BOHRPERA)

        #Reloads a previous network if name variable is not None
        if name !=  None:
            self.name = name
            self.load_network()
            LOGGER.info("Reloaded network from %s", self.network_directory)
            return

        #Data parameters
        self.molecule_set = molecule_set
        self.molecule_set_name = self.molecule_set.name
        self.elements = self.molecule_set.AtomTypes()
        self.max_num_atoms = self.molecule_set.MaxNAtoms()
        self.num_molecules = len(self.molecule_set.mols)
        self.network_type = "BehlerParinelloDirectGauSH"
        self.name = self.network_type+"_"+self.molecule_set_name+"_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")
        self.network_directory = './networks/'+self.name

        LOGGER.info("learning rate: %f", self.learning_rate)
        LOGGER.info("batch size:    %d", self.batch_size)
        LOGGER.info("max steps:     %d", self.max_steps)
        return

    def assign_activation(self):
        LOGGER.debug("Assigning Activation Function: %s", PARAMS["NeuronType"])
        try:
            if self.activation_function_type == "relu":
                self.activation_function = tf.nn.relu
            elif self.activation_function_type == "elu":
                self.activation_function = tf.nn.elu
            elif self.activation_function_type == "selu":
                self.activation_function = self.selu
            elif self.activation_function_type == "softplus":
                self.activation_function = tf.nn.softplus
            elif self.activation_function_type == "tanh":
                self.activation_function = tf.tanh
            elif self.activation_function_type == "sigmoid":
                self.activation_function = tf.sigmoid
            elif self.activation_function_type == "sigmoid_with_param":
                self.activation_function = self.sigmoid_with_param
            else:
                print ("unknown activation function, set to relu")
                self.activation_function = tf.nn.relu
        except Exception as Ex:
            print(Ex)
            print ("activation function not assigned, set to relu")
            self.activation_function = tf.nn.relu
        return

    def save_checkpoint(self, step):
        checkpoint_file = os.path.join(self.network_directory,self.name+'-checkpoint-'+str(step))
        LOGGER.info("Saving checkpoint file %s", checkpoint_file)
        self.saver.save(self.sess, checkpoint_file)
        return

    def save_network(self):
        print("Saving TFInstance")
        self.clean_scratch()
        self.clean()
        f = open(self.network_directory+".tfn","wb")
        pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        return

    def load_network(self):
        LOGGER.info("Loading TFInstance")
        f = open(self.path+"/"+self.name+".tfn","rb")
        import TensorMol.PickleTM
        network_member_variables = TensorMol.PickleTM.UnPickleTM(f)
        # self.clean()
        self.__dict__.update(network_member_variables)
        f.close()
        checkpoint_files = [x for x in os.listdir(self.network_directory) if (x.count('checkpoint')>0 and x.count('meta')==0)]
        if (len(checkpoint_files)>0):
            self.latest_checkpoint_file = checkpoint_files[0]
        else:
            LOGGER.error("Network not found in directory: %s", self.network_directory)
            LOGGER.error("Network directory contents: %s", str(os.listdir(self.network_directory)))
        return

    def clean_scratch(self):
        self.scratch_state = None
        self.scratch_pointer = 0 # for non random batch iteration.
        self.scratch_train_inputs = None
        self.scratch_train_outputs = None
        self.scratch_test_inputs = None # These should be partitioned out by LoadElementToScratch
        self.scratch_test_outputs = None
        self.molecule_set = None
        self.xyz_data = None
        self.Z_data = None
        self.energy_data = None
        self.gradient_data = None
        self.num_atoms_data = None
        return

    def reload_set(self):
        """
        Recalls the MSet to build training data etc.
        """
        self.molecule_set = MSet(self.molecule_set_name)
        self.molecule_set.Load()
        return

    def load_data(self):
        if (self.molecule_set == None):
            try:
                self.reload_set()
            except Exception as Ex:
                print("TensorData object has no molecule set.", Ex)
        if self.randomize_data:
            random.shuffle(self.molecule_set.mols)
        xyzs = np.zeros((self.num_molecules, self.max_num_atoms, 3), dtype = np.float32)
        Zs = np.zeros((self.num_molecules, self.max_num_atoms), dtype = np.int32)
        num_atoms = np.zeros((self.num_molecules), dtype = np.int32)
        energies = np.zeros((self.num_molecules), dtype = np.float32)
        dipoles = np.zeros((self.num_molecules, 3), dtype = np.float32)
        gradients = np.zeros((self.num_molecules, self.max_num_atoms, 3), dtype=np.float32)
        for i, mol in enumerate(self.molecule_set.mols):
            xyzs[i][:mol.NAtoms()] = mol.coords
            Zs[i][:mol.NAtoms()] = mol.atoms
            energies[i] = mol.properties["atomization"]
            dipoles[i] = mol.properties["dipole"]
            num_atoms[i] = mol.NAtoms()
            gradients[i][:mol.NAtoms()] = mol.properties["gradients"]
        return xyzs, Zs, energies, dipoles, num_atoms, gradients

    def load_data_to_scratch(self):
        """
        Reads built training data off disk into scratch space.
        Divides training and test data.
        Normalizes inputs and outputs.
        note that modifies my MolDigester to incorporate the normalization
        Initializes pointers used to provide training batches.

        Args:
            random: Not yet implemented randomization of the read data.

        Note:
            Also determines mean stoichiometry
        """
        self.xyz_data, self.Z_data, self.energy_data, self.dipole_data, self.num_atoms_data, self.gradient_data = self.load_data()
        self.num_test_cases = int(self.test_ratio * self.num_molecules)
        self.last_train_case = int(self.num_molecules - self.num_test_cases)
        self.num_train_cases = self.last_train_case
        self.test_scratch_pointer = self.last_train_case
        self.train_scratch_pointer = 0
        LOGGER.debug("Number of training cases: %i", self.num_train_cases)
        LOGGER.debug("Number of test cases: %i", self.num_test_cases)
        return

    def get_dipole_train_batch(self, batch_size):
        if batch_size > self.num_train_cases:
            raise Exception("Insufficent training data to fill a training batch.\n"\
                    +str(self.num_train_cases)+" cases in dataset with a batch size of "+str(batch_size))
        if self.train_scratch_pointer + batch_size >= self.num_train_cases:
            self.train_scratch_pointer = 0
        self.train_scratch_pointer += batch_size
        xyzs = self.xyz_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        Zs = self.Z_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        dipoles = self.dipole_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        gradients = self.gradient_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        num_atoms = self.num_atoms_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        return [xyzs, Zs, dipoles, gradients, num_atoms]

    def get_energy_train_batch(self, batch_size):
        if batch_size > self.num_train_cases:
            raise Exception("Insufficent training data to fill a training batch.\n"\
                    +str(self.num_train_cases)+" cases in dataset with a batch size of "+str(batch_size))
        if self.train_scratch_pointer + batch_size >= self.num_train_cases:
            self.train_scratch_pointer = 0
        self.train_scratch_pointer += batch_size
        xyzs = self.xyz_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        Zs = self.Z_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        energies = self.energy_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        dipoles = self.dipole_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        num_atoms = self.num_atoms_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        gradients = self.gradient_data[self.train_scratch_pointer - batch_size:self.train_scratch_pointer]
        NLEE = NeighborListSet(xyzs, num_atoms, False, False,  None)
        rad_eep = NLEE.buildPairs(self.Ree_cut)
        return [xyzs, Zs, energies, gradients, dipoles, num_atoms, rad_eep]

    def get_dipole_test_batch(self, batch_size):
        if batch_size > self.num_test_cases:
            raise Exception("Insufficent training data to fill a test batch.\n"\
                    +str(self.num_test_cases)+" cases in dataset with a batch size of "+str(num_cases_batch))
        if self.test_scratch_pointer + batch_size >= self.num_train_cases:
            self.test_scratch_pointer = self.last_train_case
        self.test_scratch_pointer += batch_size
        xyzs = self.xyz_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        Zs = self.Z_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        energies = self.energy_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        dipoles = self.dipole_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        gradients = self.gradient_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        num_atoms = self.num_atoms_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        return [xyzs, Zs, dipoles, gradients, num_atoms]

    def get_energy_test_batch(self, batch_size):
        if batch_size > self.num_test_cases:
            raise Exception("Insufficent training data to fill a test batch.\n"\
                    +str(self.num_test_cases)+" cases in dataset with a batch size of "+str(num_cases_batch))
        if self.test_scratch_pointer + batch_size >= self.num_train_cases:
            self.test_scratch_pointer = self.last_train_case
        self.test_scratch_pointer += batch_size
        xyzs = self.xyz_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        Zs = self.Z_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        energies = self.energy_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        dipoles = self.dipole_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        num_atoms = self.num_atoms_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        gradients = self.gradient_data[self.test_scratch_pointer - batch_size:self.test_scratch_pointer]
        NLEE = NeighborListSet(xyzs, num_atoms, False, False,  None)
        rad_eep = NLEE.buildPairs(self.Ree_cut)
        return [xyzs, Zs, energies, gradients, dipoles, num_atoms, rad_eep]

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def variable_with_weight_decay(self, shape, stddev, weight_decay, name = None):
        """
        Creates random tensorflow variable from a truncated normal distribution with weight decay

        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

        Returns:
            Variable Tensor

        Notes:
            Note that the Variable is initialized with a truncated normal distribution.
            A weight decay is added only if one is specified.
        """
        variable = tf.Variable(tf.truncated_normal(shape, stddev = stddev, dtype = self.tf_precision), name = name)
        if weight_decay is not None:
            weightdecay = tf.multiply(tf.nn.l2_loss(variable), weight_decay, name='weight_loss')
            tf.add_to_collection('losses', weightdecay)
        return variable

    def compute_normalization(self):
        xyzs_pl = tf.placeholder(self.tf_precision, shape=tuple([None, self.max_num_atoms, 3]))
        Zs_pl = tf.placeholder(tf.int32, shape=tuple([None, self.max_num_atoms]))
        gaussian_params = tf.Variable(self.gaussian_params, trainable=False, dtype=self.tf_precision)
        elements = tf.constant(self.elements, dtype = tf.int32)

        rotation_params = tf.stack([np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision),
                np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision),
                tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision)], axis=-1, name="rotation_params")
        rotated_xyzs = tf_random_rotate(xyzs_pl, rotation_params)
        embeddings, molecule_indices = tf_gaussian_spherical_harmonics_channel(rotated_xyzs, Zs_pl, elements, gaussian_params, self.l_max)

        embeddings_list = []
        for element in range(len(self.elements)):
            embeddings_list.append([])
        labels_list = []
        self.embeddings_mean = []
        self.embeddings_stddev = []

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for ministep in range (0, max(2, int(0.1 * self.num_train_cases/self.batch_size))):
            batch_data = self.get_energy_train_batch(self.batch_size)
            labels_list.append(batch_data[2])
            embedding, molecule_index = sess.run([embeddings, molecule_indices], feed_dict = {xyzs_pl:batch_data[0], Zs_pl:batch_data[1]})
            for element in range(len(self.elements)):
                embeddings_list[element].append(embedding[element])
        sess.close()
        for element in range(len(self.elements)):
            self.embeddings_mean.append(np.mean(np.concatenate(embeddings_list[element]), axis=0))
            self.embeddings_stddev.append(np.std(np.concatenate(embeddings_list[element]), axis=0))
        self.embeddings_mean = np.stack(self.embeddings_mean)
        self.embeddings_stddev = np.stack(self.embeddings_stddev)
        labels = np.concatenate(labels_list)
        self.labels_mean = np.mean(labels)
        self.labels_stddev = np.std(labels)
        self.train_scratch_pointer = 0

        #Set the embedding and label shape
        self.embedding_shape = embedding[0].shape[1]
        self.label_shape = labels[0].shape
        return

    def clean(self):
        self.sess = None
        self.xyzs_pl = None
        self.Zs_pl = None
        self.labels_pl = None
        self.dipole_pl = None
        self.gradients_pl = None
        self.num_atoms_pl = None
        self.Reep_pl = None
        self.output = None
        self.gradients = None
        self.gradient_labels = None
        self.dipoles = None
        self.charges = None
        self.net_charge = None
        self.dipole_labels = None
        self.coulomb_energy = None
        self.total_energy = None
        self.energy_losses = None
        self.dipole_losses = None
        self.energy_loss = None
        self.dipole_loss = None
        self.gradient_loss = None
        self.charge_loss = None
        self.dipole_train_op = None
        self.energy_train_op = None
        self.saver = None
        self.summary_writer = None
        self.summary_op = None
        self.activation_function = None
        self.options = None
        self.run_metadata = None
        self.gaussian_params = None
        return

    def train_prepare(self,  continue_training =False):
        """
        Get placeholders, graph and losses in order to begin training.
        Also assigns the desired padding.

        Args:
            continue_training: should read the graph variables from a saved checkpoint.
        """
        with tf.Graph().as_default():
            #Define the placeholders to be fed in for each batch
            self.xyzs_pl = tf.placeholder(self.tf_precision, shape=[None, self.max_num_atoms, 3])
            self.Zs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms])
            self.labels_pl = tf.placeholder(self.tf_precision, shape=[None])
            self.dipole_pl = tf.placeholder(self.tf_precision, shape=[None, 3])
            self.gradients_pl = tf.placeholder(self.tf_precision, shape=[None, self.max_num_atoms, 3])
            self.num_atoms_pl = tf.placeholder(tf.int32, shape=[None])
            self.Reep_pl = tf.placeholder(tf.int32, shape=[None,3])

            #Define the embedding parameters and normalization constants
            self.gaussian_params = tf.Variable(self.gaussian_params, trainable=True, dtype=self.tf_precision)
            elements = tf.Variable(self.elements, trainable=False, dtype = tf.int32)
            embeddings_mean = tf.Variable(self.embeddings_mean, trainable=False, dtype = self.tf_precision)
            embeddings_stddev = tf.Variable(self.embeddings_stddev, trainable=False, dtype = self.tf_precision)
            labels_mean = tf.Variable(self.labels_mean, trainable=False, dtype = self.tf_precision)
            labels_stddev = tf.Variable(self.labels_stddev, trainable=False, dtype = self.tf_precision)
            elu_width = tf.Variable(self.elu_width * BOHRPERA, trainable=False, dtype = self.tf_precision)
            elu_alpha = tf.Variable(self.elu_alpha, trainable=False, dtype = self.tf_precision)
            elu_shift = tf.Variable(self.elu_shift, trainable=False, dtype = self.tf_precision)
            damp_shifted_alpha = tf.Variable(self.damp_shifted_alpha, trainable=False, dtype = self.tf_precision)

            rotation_params = tf.stack([np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision),
                    np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision),
                    tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision)], axis=-1)
            rotated_xyzs, rotated_gradients = tf_random_rotate(self.xyzs_pl, rotation_params, self.gradients_pl)
            self.dipole_labels = tf.squeeze(tf_random_rotate(tf.expand_dims(self.dipole_pl, axis=1), rotation_params))
            embeddings, molecule_indices = tf_gaussian_spherical_harmonics_channel(rotated_xyzs,
                                            self.Zs_pl, elements, self.gaussian_params, self.l_max)
            for element in range(len(self.elements)):
                embeddings[element] -= embeddings_mean[element]
                embeddings[element] /= embeddings_stddev[element]

            self.dipoles, self.charges, self.net_charge, dipole_variables = self.dipole_inference(embeddings, molecule_indices, rotated_xyzs, self.num_atoms_pl)
            # dipole_variables.append(self.gaussian_params)
            self.coulomb_energy = tf_coulomb_damp_shifted_force(rotated_xyzs * BOHRPERA, self.charges, elu_width, self.Reep_pl,
                                    damp_shifted_alpha, elu_alpha, elu_shift)
            norm_output, energy_variables = self.energy_inference(embeddings, molecule_indices)
            self.output = (norm_output * labels_stddev) + labels_mean
            self.total_energy = self.output + self.coulomb_energy
            self.gradients = tf.gather_nd(tf.gradients(self.output, rotated_xyzs)[0], tf.where(tf.not_equal(self.Zs_pl, 0)))
            self.gradient_labels = tf.gather_nd(rotated_gradients, tf.where(tf.not_equal(self.Zs_pl, 0)))
            num_atoms_batch = tf.reduce_sum(self.num_atoms_pl)

            self.energy_loss = self.energy_loss_op(self.total_energy, self.labels_pl)
            self.dipole_loss = self.dipole_loss_op(self.dipoles, self.dipole_labels)
            # self.charge_loss = self.charge_loss_op(self.net_charge)
            if self.train_energy_gradients:
                self.gradient_loss = self.gradient_loss_op(self.gradients, self.gradient_labels, num_atoms_batch)
            self.dipole_losses = tf.add_n(tf.get_collection('dipole_losses'))
            self.energy_losses = tf.add_n(tf.get_collection('energy_losses'))

            # barrier_function = -1000.0 * tf.log(tf.concat([self.gaussian_params + 0.9,
            #                     tf.expand_dims(6.5 - self.gaussian_params[:,0], axis=-1),
            #                     tf.expand_dims(1.75 - self.gaussian_params[:,1], axis=-1)], axis=1))
            # truncated_barrier_function = tf.reduce_sum(tf.where(tf.greater(barrier_function, 0.0),
            #                     barrier_function, tf.zeros_like(barrier_function)))
            # gaussian_overlap_loss = tf.square(0.001 / tf.reduce_min(tf.self_adjoint_eig(tf_gaussian_overlap(self.gaussian_params))[0]))
            # loss_and_constraint = self.total_loss + truncated_barrier_function + gaussian_overlap_loss

            self.dipole_train_op = self.optimizer(self.dipole_losses, self.learning_rate, self.momentum, dipole_variables)
            self.energy_train_op = self.optimizer(self.energy_losses, self.learning_rate, self.momentum, energy_variables)
            self.summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
            self.summary_writer = tf.summary.FileWriter(self.network_directory, self.sess.graph)
            self.sess.run(init)
            if self.profiling:
                self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                self.run_metadata = tf.RunMetadata()
        return

    def fill_dipole_feed_dict(self, batch_data):
        """
        Fill the tensorflow feed dictionary.

        Args:
            batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
            and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

        Returns:
            Filled feed dictionary.
        """
        feed_dict={i: d for i, d in zip([self.xyzs_pl, self.Zs_pl, self.dipole_pl, self.gradients_pl, self.num_atoms_pl], batch_data)}
        return feed_dict

    def fill_energy_feed_dict(self, batch_data):
        """
        Fill the tensorflow feed dictionary.

        Args:
            batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
            and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

        Returns:
            Filled feed dictionary.
        """
        feed_dict={i: d for i, d in zip([self.xyzs_pl, self.Zs_pl, self.labels_pl, self.gradients_pl, self.dipole_pl, self.num_atoms_pl, self.Reep_pl], batch_data)}
        return feed_dict

    def energy_inference(self, inp, indexs):
        """
        Builds a Behler-Parinello graph

        Args:
            inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
            index: a list of (num_of atom type X batchsize) array which linearly combines the elements
        Returns:
            The BP graph output
        """
        branches=[]
        variables=[]
        output = tf.zeros([self.batch_size, self.max_num_atoms], dtype=self.tf_precision)
        with tf.name_scope("energy_network"):
            for e in range(len(self.elements)):
                branches.append([])
                inputs = inp[e]
                index = indexs[e]
                for i in range(len(self.hidden_layers)):
                    if i == 0:
                        with tf.name_scope(str(self.elements[e])+'_hidden1'):
                            weights = self.variable_with_weight_decay(shape=[self.embedding_shape, self.hidden_layers[i]],
                                    stddev=math.sqrt(2.0 / float(self.embedding_shape)), weight_decay=self.weight_decay, name="weights")
                            biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
                            branches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
                            variables.append(weights)
                            variables.append(biases)
                    else:
                        with tf.name_scope(str(self.elements[e])+'_hidden'+str(i+1)):
                            weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
                                    stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
                            biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
                            branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
                            variables.append(weights)
                            variables.append(biases)
                with tf.name_scope(str(self.elements[e])+'_regression_linear'):
                    weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], 1],
                            stddev=math.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
                    biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
                    branches[-1].append(tf.squeeze(tf.matmul(branches[-1][-1], weights) + biases))
                    variables.append(weights)
                    variables.append(biases)
                    output += tf.scatter_nd(index, branches[-1][-1], [self.batch_size, self.max_num_atoms])
                tf.verify_tensor_all_finite(output,"Nan in output!!!")
        return tf.reshape(tf.reduce_sum(output, axis=1), [self.batch_size]), variables

    def dipole_inference(self, inp, indexs, xyzs, natom):
        """
        Builds a Behler-Parinello graph

        Args:
            inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
            index: a list of (num_of atom type X batchsize) array which linearly combines the elements
        Returns:
            The BP graph output
        """
        xyzs *= BOHRPERA
        branches=[]
        variables=[]
        atom_outputs_charge = []
        output = tf.zeros([self.batch_size, self.max_num_atoms], dtype=self.tf_precision)
        with tf.name_scope("dipole_network"):
            for e in range(len(self.elements)):
                branches.append([])
                inputs = inp[e]
                index = indexs[e]
                for i in range(len(self.hidden_layers)):
                    if i == 0:
                        with tf.name_scope(str(self.elements[e])+'_hidden1'):
                            weights = self.variable_with_weight_decay(shape=[self.embedding_shape, self.hidden_layers[i]],
                                    stddev=math.sqrt(2.0 / float(self.embedding_shape)), weight_decay=self.weight_decay, name="weights")
                            biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
                            branches[-1].append(self.activation_function(tf.matmul(inputs, weights) + biases))
                            variables.append(weights)
                            variables.append(biases)
                    else:
                        with tf.name_scope(str(self.elements[e])+'_hidden'+str(i+1)):
                            weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
                                    stddev=math.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
                            biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
                            branches[-1].append(self.activation_function(tf.matmul(branches[-1][-1], weights) + biases))
                            variables.append(weights)
                            variables.append(biases)
                with tf.name_scope(str(self.elements[e])+'_regression_linear'):
                    weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], 1],
                            stddev=math.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
                    biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
                    branches[-1].append(tf.squeeze(tf.matmul(branches[-1][-1], weights) + biases))
                    variables.append(weights)
                    variables.append(biases)
                    output += tf.scatter_nd(index, branches[-1][-1], [self.batch_size, self.max_num_atoms])

            tf.verify_tensor_all_finite(output,"Nan in output!!!")
            net_charge = tf.reduce_sum(output, axis=1)
            delta_charge = net_charge / tf.cast(natom, self.tf_precision)
            charges = output - tf.expand_dims(delta_charge, axis=1)
            dipole = tf.reduce_sum(xyzs * tf.expand_dims(charges, axis=-1), axis=1)
        return dipole, charges, net_charge, variables

    def optimizer(self, loss, learning_rate, momentum, variables):
        """
        Sets up the training Ops.
        Creates a summarizer to track the loss over time in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
            loss: Loss tensor, from loss().
            learning_rate: the learning rate to use for gradient descent.

        Returns:
            train_op: the tensorflow operation to call for training.
        """
        tf.summary.scalar(loss.op.name, loss)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step, var_list=variables)
        return train_op

    def energy_loss_op(self, total_energies, total_energy_labels):
        energy_loss = tf.nn.l2_loss(total_energies - total_energy_labels)
        tf.add_to_collection('energy_losses', energy_loss)
        return energy_loss

    def gradient_loss_op(self, gradients, gradient_labels, num_atoms):
        gradient_loss = tf.nn.l2_loss(gradients - gradient_labels) / tf.cast(num_atoms, self.tf_precision)
        tf.add_to_collection('energy_losses', gradient_loss)
        return gradient_loss

    def dipole_loss_op(self, dipoles, dipole_labels):
        dipole_loss = tf.nn.l2_loss(dipoles - dipole_labels)
        tf.add_to_collection('dipole_losses', dipole_loss)
        return dipole_loss

    def charge_loss_op(self, net_charge):
        charge_loss = tf.nn.l2_loss(net_charge)
        tf.add_to_collection('dipole_losses', charge_loss)
        return charge_loss

    def dipole_train_step(self, step):
        """
        Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

        Args:
            step: the index of this step.
        """
        Ncase_train = self.num_train_cases
        start_time = time.time()
        train_loss =  0.0
        train_energy_loss = 0.0
        train_dipole_loss = 0.0
        # train_charge_loss = 0.0
        train_gradient_loss = 0.0
        num_mols = 0
        for ministep in range (0, int(Ncase_train/self.batch_size)):
            batch_data = self.get_dipole_train_batch(self.batch_size)
            feed_dict = self.fill_dipole_feed_dict(batch_data)
            _, total_loss, dipole_loss = self.sess.run([self.dipole_train_op, self.dipole_losses, self.dipole_loss], feed_dict=feed_dict)
            train_loss += total_loss
            train_dipole_loss += dipole_loss
            # train_charge_loss += charge_loss
            num_mols += self.batch_size
        duration = time.time() - start_time
        LOGGER.info("step: %5d    duration: %10.5f  train loss: %13.10f  dipole loss: %13.10f", step, duration,
            train_loss / num_mols, train_dipole_loss / num_mols)
        return

    def energy_train_step(self, step):
        """
        Perform a single training step (complete processing of all input), using minibatches of size self.batch_size

        Args:
            step: the index of this step.
        """
        Ncase_train = self.num_train_cases
        start_time = time.time()
        train_loss =  0.0
        train_energy_loss = 0.0
        train_gradient_loss = 0.0
        num_mols = 0
        for ministep in range (0, int(Ncase_train/self.batch_size)):
            batch_data = self.get_energy_train_batch(self.batch_size)
            feed_dict = self.fill_energy_feed_dict(batch_data)
            if self.train_energy_gradients:
                _, total_loss, energy_loss, gradient_loss = self.sess.run([self.energy_train_op,
                self.energy_losses, self.energy_loss, self.gradient_loss], feed_dict=feed_dict)
                train_gradient_loss += gradient_loss
            else:
                _, total_loss, energy_loss = self.sess.run([self.energy_train_op,
                self.energy_losses, self.energy_loss], feed_dict=feed_dict)
            train_loss += total_loss
            train_energy_loss += energy_loss
            num_mols += self.batch_size
        duration = time.time() - start_time
        if self.train_energy_gradients:
            self.print_training(step, train_loss, train_energy_loss, num_mols, duration, train_gradient_loss)
        else:
            self.print_training(step, train_loss, train_energy_loss, num_mols, duration)
        return

    def dipole_test_step(self, step):
        """
        Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

        Args:
            step: the index of this step.
        """
        print( "testing...")
        test_loss =  0.0
        start_time = time.time()
        Ncase_test = self.num_test_cases
        num_mols = 0
        test_loss = 0.0
        test_dipole_loss = 0.0
        # test_charge_loss = 0.0
        test_epoch_dipole_labels, test_epoch_dipole_outputs = [], []
        test_net_charges = []
        for ministep in range (0, int(Ncase_test/self.batch_size)):
            batch_data = self.get_dipole_test_batch(self.batch_size)
            feed_dict = self.fill_dipole_feed_dict(batch_data)
            dipoles, dipole_labels, total_loss, dipole_loss, gaussian_params = self.sess.run([self.dipoles, self.dipole_labels,
            self.dipole_losses, self.dipole_loss, self.gaussian_params],  feed_dict=feed_dict)
            num_mols += self.batch_size
            test_loss += total_loss
            test_dipole_loss += dipole_loss
            # test_charge_loss += charge_loss
            test_epoch_dipole_labels.append(dipole_labels)
            test_epoch_dipole_outputs.append(dipoles)
            # test_net_charges.append(net_charges)
        test_epoch_dipole_labels = np.concatenate(test_epoch_dipole_labels)
        test_epoch_dipole_outputs = np.concatenate(test_epoch_dipole_outputs)
        test_epoch_dipole_errors = test_epoch_dipole_labels - test_epoch_dipole_outputs
        # test_net_charges = np.concatenate(test_net_charges)
        duration = time.time() - start_time
        # for i in [random.randint(0, self.batch_size - 1) for _ in xrange(20)]:
        #     LOGGER.info("Net Charges: %11.8f", test_net_charges[i])
        for i in [random.randint(0, self.batch_size - 1) for _ in xrange(20)]:
            LOGGER.info("Dipole label: %s  Dipole output: %s", test_epoch_dipole_labels[i], test_epoch_dipole_outputs[i])
        # LOGGER.info("MAE  Dipole: %11.8f  Net Charge: %11.8f", np.mean(np.abs(test_epoch_dipole_errors)), np.mean(np.abs(test_net_charges)))
        # LOGGER.info("MSE  Dipole: %11.8f  Net Charge: %11.8f", np.mean(test_epoch_dipole_errors), np.mean(test_net_charges))
        # LOGGER.info("RMSE Dipole: %11.8f  Net Charge: %11.8f", np.sqrt(np.mean(np.square(test_epoch_dipole_errors))), np.sqrt(np.mean(np.square(test_net_charges))))
        LOGGER.info("MAE  Dipole: %11.8f", np.mean(np.abs(test_epoch_dipole_errors)))
        LOGGER.info("MSE  Dipole: %11.8f", np.mean(test_epoch_dipole_errors))
        LOGGER.info("RMSE Dipole: %11.8f", np.sqrt(np.mean(np.square(test_epoch_dipole_errors))))
        LOGGER.info("Gaussian paramaters: %s", gaussian_params)
        LOGGER.info("step: %5d    duration: %10.5f   test loss: %13.10f  dipole loss: %13.10f", step, duration,
            test_loss / num_mols, test_dipole_loss / num_mols)
        return test_loss

    def energy_test_step(self, step):
        """
        Perform a single test step (complete processing of all input), using minibatches of size self.batch_size

        Args:
            step: the index of this step.
        """
        print( "testing...")
        test_loss =  0.0
        start_time = time.time()
        Ncase_test = self.num_test_cases
        num_mols = 0
        test_energy_loss = 0.0
        test_gradient_loss = 0.0
        test_charge_loss = 0.0
        test_epoch_energy_labels, test_epoch_energy_outputs = [], []
        test_epoch_force_labels, test_epoch_force_outputs = [], []
        num_atoms_epoch = []
        for ministep in range (0, int(Ncase_test/self.batch_size)):
            batch_data = self.get_energy_test_batch(self.batch_size)
            feed_dict = self.fill_energy_feed_dict(batch_data)
            if self.train_energy_gradients:
                total_energies, energy_labels, gradients, gradient_labels, total_loss, energy_loss, gradient_loss, num_atoms, gaussian_params = self.sess.run([self.total_energy,
                self.labels_pl, self.gradients, self.gradient_labels, self.energy_losses, self.energy_loss,
                self.gradient_loss, self.num_atoms_pl, self.gaussian_params],  feed_dict=feed_dict)
                test_gradient_loss += gradient_loss
            else:
                total_energies, energy_labels, gradients, gradient_labels, total_loss, energy_loss, num_atoms, gaussian_params = self.sess.run([self.total_energy,
                self.labels_pl, self.gradients, self.gradient_labels, self.energy_losses, self.energy_loss, self.num_atoms_pl, self.gaussian_params],  feed_dict=feed_dict)
            test_loss += total_loss
            num_mols += self.batch_size
            test_energy_loss += energy_loss
            test_epoch_energy_labels.append(energy_labels)
            test_epoch_energy_outputs.append(total_energies)
            test_epoch_force_labels.append(-1.0 * gradient_labels)
            test_epoch_force_outputs.append(-1.0 * gradients)
            num_atoms_epoch.append(num_atoms)
        test_epoch_energy_labels = np.concatenate(test_epoch_energy_labels)
        test_epoch_energy_outputs = np.concatenate(test_epoch_energy_outputs)
        test_epoch_energy_errors = test_epoch_energy_labels - test_epoch_energy_outputs
        test_epoch_force_labels = np.concatenate(test_epoch_force_labels)
        test_epoch_force_outputs = np.concatenate(test_epoch_force_outputs)
        test_epoch_force_errors = test_epoch_force_labels - test_epoch_force_outputs
        num_atoms_epoch = np.sum(np.concatenate(num_atoms_epoch))
        duration = time.time() - start_time
        for i in [random.randint(0, self.batch_size - 1) for _ in xrange(20)]:
            LOGGER.info("Energy label: %11.8f  Energy output: %11.8f", test_epoch_energy_labels[i], test_epoch_energy_outputs[i])
        for i in [random.randint(0, num_atoms_epoch - 1) for _ in xrange(20)]:
            LOGGER.info("Forces label: %s  Forces output: %s", test_epoch_force_labels[i], test_epoch_force_outputs[i])
        LOGGER.info("MAE  Energy: %11.8f  Forces: %11.8f", np.mean(np.abs(test_epoch_energy_errors)),
        np.mean(np.abs(test_epoch_force_errors)))
        LOGGER.info("MSE  Energy: %11.8f  Forces: %11.8f", np.mean(test_epoch_energy_errors),
        np.mean(test_epoch_force_errors))
        LOGGER.info("RMSE Energy: %11.8f  Forces: %11.8f", np.sqrt(np.mean(np.square(test_epoch_energy_errors))),
        np.sqrt(np.mean(np.square(test_epoch_force_errors))))
        LOGGER.info("Gaussian paramaters: %s", gaussian_params)
        if self.train_energy_gradients:
            self.print_testing(step, test_loss, test_energy_loss, num_mols, duration, test_gradient_loss)
        else:
            self.print_testing(step, test_loss, test_energy_loss, num_mols, duration)
        return test_loss

    def train(self):
        self.load_data_to_scratch()
        self.compute_normalization()
        self.train_prepare()
        test_freq = PARAMS["test_freq"]
        mini_test_loss = 1e10
        for step in range(1, 151):
            self.dipole_train_step(step)
            if step%test_freq==0:
                test_loss = self.dipole_test_step(step)
                if (test_loss < mini_test_loss):
                    mini_test_loss = test_loss
                    self.save_checkpoint(step)
        LOGGER.info("Continue training dipole until new best checkpoint found.")
        train_energy_flag = False
        step += 1
        while train_energy_flag == False:
            self.dipole_train_step(step)
            test_loss = self.dipole_test_step(step)
            if (test_loss < mini_test_loss):
                mini_test_loss = test_loss
                self.save_checkpoint(step)
                train_energy_flag=True
                LOGGER.info("New best checkpoint found. Start training energy network.")
            step += 1
        mini_test_loss = 1e10
        for step in range(1, self.max_steps+1):
            self.energy_train_step(step)
            if step%test_freq==0:
                test_loss = self.energy_test_step(step)
                if (test_loss < mini_test_loss):
                    mini_test_loss = test_loss
                    self.save_checkpoint(step)
        self.sess.close()
        self.save_network()
        return

    def print_training(self, step, loss, energy_loss, num_mols, duration, gradient_loss=None):
        if self.train_energy_gradients:
            LOGGER.info("step: %5d    duration: %.5f  train loss: %.10f  energy loss: %.10f  gradient loss: %.10f",
            step, duration, loss / num_mols, energy_loss / num_mols, gradient_loss / num_mols)
        else:
            LOGGER.info("step: %5d    duration: %.5f  train loss: %.10f energy loss: %.10f",
            step, duration, loss / num_mols, energy_loss / num_mols)
        return

    def print_testing(self, step, loss, energy_loss, num_mols, duration, gradient_loss=None):
        if self.train_energy_gradients:
            LOGGER.info("step: %5d    duration: %.5f  test loss: %.10f  energy loss: %.10f  gradient loss: %.10f",
            step, duration, loss / num_mols, energy_loss / num_mols, gradient_loss / num_mols)
        else:
            LOGGER.info("step: %5d    duration: %.5f  test loss: %.10f energy loss: %.10f",
            step, duration, loss / num_mols, energy_loss / num_mols)
        return

    def evaluate_prepare(self):
        """
        Get placeholders, graph and losses in order to begin training.
        Also assigns the desired padding.

        Args:
            continue_training: should read the graph variables from a saved checkpoint.
        """
        with tf.Graph().as_default():
            self.xyzs_pl = tf.placeholder(self.tf_precision, shape=tuple([None, self.num_atoms, 3]))
            self.Zs_pl = tf.placeholder(tf.int32, shape=tuple([None, self.num_atoms]))

            self.gaussian_params = tf.Variable(self.gaussian_params, trainable=False, dtype=self.tf_precision)
            elements = tf.Variable(self.elements, trainable=False, dtype = tf.int32)
            embeddings_mean = tf.Variable(self.embeddings_mean, trainable=False, dtype = self.tf_precision)
            embeddings_stddev = tf.Variable(self.embeddings_stddev, trainable=False, dtype = self.tf_precision)
            labels_mean = tf.Variable(self.labels_mean, trainable=False, dtype = self.tf_precision)
            labels_stddev = tf.Variable(self.labels_stddev, trainable=False, dtype = self.tf_precision)

            self.tiled_xyzs = tf.tile(tf.expand_dims(self.xyzs_pl, axis=1), [1, 100, 1, 1])
            self.tiled_Zs = tf.tile(tf.expand_dims(self.Zs_pl, axis=1), [1, 100, 1])
            self.rotation_params = tf.tile(tf.expand_dims(tf.concat([tf.expand_dims(tf.tile(tf.linspace(0.001, 1.999, 5), [20]), axis=1),
                    tf.reshape(tf.tile(tf.expand_dims(tf.linspace(0.001, 1.999, 5), axis=1), [1,20]), [100,1]),
                    tf.reshape(tf.tile(tf.expand_dims(tf.expand_dims(tf.linspace(0.001, 1.999, 4), axis=1),
                    axis=2), [5,1,5]), [100,1])], axis=1), axis=0), [self.batch_size, 1, 1])
            self.rotated_xyzs = tf_random_rotate(self.tiled_xyzs, self.rotation_params)
            self.rotated_xyzs = tf.reshape(self.rotated_xyzs, [-1, self.num_atoms, 3])
            # self.rotated_gradients = tf.reshape(rotated_gradients, [-1, self.max_num_atoms, 3])
            self.tiled_Zs = tf.reshape(self.tiled_Zs, [-1, self.num_atoms])

            # tiled_xyzs = tf.tile(self.xyzs_pl, [self.batch_size, 1, 1])
            # tiled_Zs = tf.tile(self.Zs_pl, [self.batch_size, 1])
            # rotation_params = tf.concat([np.pi * tf.expand_dims(tf.tile(tf.linspace(0.001, 1.999, 5), [20]), axis=1),
            #         np.pi * tf.reshape(tf.tile(tf.expand_dims(tf.linspace(0.001, 1.999, 5), axis=1), [1,20]), [100,1]),
            #         tf.reshape(tf.tile(tf.expand_dims(tf.expand_dims(tf.linspace(0.001, 1.999, 4), axis=1),
            #         axis=2), [5,1,5]), [100,1])], axis=1)
            # rotation_params = tf.stack([np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision),
            #         np.pi * tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision),
            #         tf.random_uniform([self.batch_size], maxval=2.0, dtype=self.tf_precision)], axis=-1, name="rotation_params")
            # rotated_xyzs = tf_random_rotate(tiled_xyzs, rotation_params)
            # embeddings, molecule_indices = tf_gaussian_spherical_harmonics_channel(rotated_xyzs,
            #                                 tiled_Zs, elements, self.gaussian_params, self.l_max)
            # for element in range(len(self.elements)):
            #     embeddings[element] -= embeddings_mean[element]
            #     embeddings[element] /= embeddings_stddev[element]
            #
            # norm_output = self.inference(embeddings, molecule_indices)
            # self.output = (norm_output * labels_stddev) + labels_mean
            #
            # self.gradients = tf.gradients(self.output, self.xyzs_pl)[0] / self.batch_size

            self.summary_op = tf.summary.merge_all()
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
            # self.summary_writer = tf.summary.FileWriter(self.network_directory, self.sess.graph)
        return

    def evaluate_fill_feed_dict(self, xyzs, Zs):
        """
        Fill the tensorflow feed dictionary.

        Args:
            batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
            and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

        Returns:
            Filled feed dictionary.
        """
        feed_dict={i: d for i, d in zip([self.xyzs_pl, self.Zs_pl], [xyzs, Zs])}
        return feed_dict

    def evaluate_mol(self, mol, eval_forces=True):
        """
        Takes coordinates and atomic numbers from a manager and feeds them into the network
        for evaluation of the forces

        Args:
            xyzs (np.float): numpy array of atomic coordinates
            Zs (np.int32): numpy array of atomic numbers
        """
        if (mol.NAtoms() > self.max_num_atoms):
            self.max_num_atoms = mol.NAtoms()
            self.gaussian_params = PARAMS["RBFS"][:self.number_radial]
            self.assign_activation()
            self.num_atoms = self.max_num_atoms
            self.evaluate_prepare()
        elif not self.sess:
            print("loading the session..")
            self.gaussian_params = PARAMS["RBFS"][:self.number_radial]
            self.assign_activation()
            self.num_atoms = mol.NAtoms()
            self.batch_size = 1
            self.evaluate_prepare()
        xyzs_feed = np.zeros((1,self.num_atoms, 3))
        Zs_feed = np.zeros((1,self.num_atoms))
        xyzs_feed[0,:mol.NAtoms()] = mol.coords
        Zs_feed[0,:mol.NAtoms()] = mol.atoms
        feed_dict=self.evaluate_fill_feed_dict(xyzs_feed, Zs_feed)
        energy = self.sess.run(self.rotated_xyzs, feed_dict=feed_dict)
        return energy
        # atomization_energy = 0.0
        # for atom in mol.atoms:
        #     if atom in ele_U:
        #         atomization_energy += ele_U[atom]
        # if eval_forces:
        #     energy, gradients = self.sess.run([self.output, self.gradients], feed_dict=feed_dict)
        #     forces = -gradients[0,:mol.NAtoms()]
        #     return np.mean(energy) + atomization_energy, forces
        # else:
        #     energy = self.sess.run(self.output, feed_dict=feed_dict)
        #     return np.mean(energy) + atomization_energy
