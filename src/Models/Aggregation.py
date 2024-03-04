# Deep learning
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import Sequential, Model, layers
from tensorflow.keras.layers import Dense, concatenate, Flatten,MultiHeadAttention
import tensorflow.keras.backend as K
# The above code is a Python code snippet that is commented out. It appears to be importing the
# `layers` module from `tensorflow.keras` library, but the import statement is commented out with `#`.
# This means that the import statement is not active and will not be executed when the code is run.
# from tensorflow.keras import layers
import matplotlib.pyplot as plt

""" Create the Model """
def smape(y_true, y_pred):
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    return smape

class Attn_Net_Gated(tf.keras.Model):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net_Gated, self).__init__()

        self.attention_a = [
            tf.keras.layers.Dense(D),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dropout(dropout)
        ]

        self.attention_b = [
            tf.keras.layers.Dense(D),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dropout(dropout)
        ]

        self.attention_a = tf.keras.Sequential(self.attention_a)
        self.attention_b = tf.keras.Sequential(self.attention_b)

        self.attention_c = tf.keras.layers.Dense(n_classes)

    def call(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = tf.multiply(a, b)
        A = self.attention_c(A)
        return A

class Fusion_Net(tf.keras.Model):
    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Fusion_Net, self).__init__()

        self.attention_a = [
            tf.keras.layers.Dense(D),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Dropout(dropout)
        ]

        self.attention_b = [
            tf.keras.layers.Dense(D),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dropout(dropout)
        ]

        self.attention_a = tf.keras.Sequential(self.attention_a)
        self.attention_b = tf.keras.Sequential(self.attention_b)

        self.attention_c = tf.keras.layers.Dense(n_classes)

    def call(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = tf.multiply(a, b)
        A = self.attention_c(A)
        return A

# Defines the Kronecker product
from tensorflow.keras.layers import Layer
class OuterProductLayer(Layer):
    def __init__(self):
        super(OuterProductLayer, self).__init__()

    def call(self, inputs):
        # Assuming inputs is a tuple of two tensors
        output1, output2 = inputs
        ones_tensor = tf.ones((tf.shape(output1)[0], 1), dtype=output1.dtype)
        ones_tensor = tf.stop_gradient(ones_tensor)
        # Concatenate the ones tensor with the input vectors
        output1 = tf.concat([ones_tensor, output1], axis=1)
        ones_tensor2 = tf.ones((tf.shape(output1)[0], 1), dtype=output1.dtype)
        ones_tensor2 = tf.stop_gradient(ones_tensor2)
        # Concatenate the ones tensor with the input vectors
        output2 = tf.concat([ones_tensor, output2], axis=1)
        outer_product = tf.einsum('bi,bj->bij', output1, output2)
        return Flatten()(outer_product)

def create_aggregation_model(model, model_2, model_3=None, fusion=None, dense_acivation='relu'):
    L=128
    D=64

    print(model.summary(), "model1 sum")

    if fusion == 'early' or 'late':
        modelc = Sequential(model.layers[:-1])
        modelc.trainable = False
        model_2c = Sequential(model_2.layers[:-1])
        model_2c.trainable = False
        if model_3:
            model_3 = Sequential(model_3.layers[:-1])
            model_3.trainable = False

    if model_3:
        input3 = model_3.input
        out_3 = model_3(input3)
    # Create a Sequential Model
    input1 = modelc.inputs
    input2 = model_2c.input
    out_1 = modelc(input1)
    out_2 = model_2c(input2)
    out_1c = model(input1)
    out_2c = model_2(input2)

    fusion_features = OuterProductLayer()((out_1,out_2))
    print(fusion_features.shape)
    out1_i = Dense(120, activation=dense_acivation)(out_1)
    out2_i = Dense(120, activation=dense_acivation)(out_2)
    out_x = Dense(120, activation=dense_acivation)(concatenate((out_1,out_2)))
    # Assuming self.fusion_fc is a Keras layer
    x = Dense(256, activation=dense_acivation)(fusion_features)

    x = Dense(120, activation=dense_acivation)(x)

    estimator_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1,activation='sigmoid')  # Outputs a scalar
        ])
    estimator_net2 = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1,activation='sigmoid')  # Outputs a scalar
        ])
    estimator_net3 = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1,activation='sigmoid')  # Outputs a scalar
        ])
    joint_estimate = estimator_net(out_x)
    out_x = Dense(10,activation=dense_acivation)(out_x)
    x = concatenate([x,out_1,out_2,out_x])
    print(x.shape,out_1.shape,out_2.shape)
    # print(intermediate.shape,"inter")
    marginal_estimate_1 = estimator_net2(out1_i)
    marginal_estimate_2 = estimator_net2(out2_i)
    
    
    x = Dense(64, activation=dense_acivation)(x)

    x = Dense(32, activation=dense_acivation)(x)

    x = Dense(16, activation= dense_acivation)(x)
    output_layer = Dense(1)(x)

    #Model Definition
    final_model = Model(inputs=[(input1, input2)], outputs=[output_layer])


    # Compile the model:
    if int(tf.__version__.split('.')[1]) >= 11:
        opt = tf.keras.optimizers.legacy.Adam(lr=0.001)
    else:
        opt = tf.keras.optimizers.Adam(lr=0.001)

    # Metrics
    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        smape
    ]

    final_model.add_loss(  -0.1*(tf.reduce_mean(joint_estimate) - tf.math.log(tf.reduce_mean(tf.exp(marginal_estimate_1))))-0.1*(tf.reduce_mean(joint_estimate) - tf.math.log(tf.reduce_mean(tf.exp(marginal_estimate_2))))  )
    # final_model.add_loss()
    final_model.compile(loss='mse', optimizer=opt, metrics=metrics)
    return final_model
avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

class custom_fit(tf.keras.Model):
    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True) # forward pass 
            reg_loss = tf.reduce_sum(self.losses)
            pred_loss = loss(labels, outputs)
            total_loss = tf.reduce_sum(pred_loss) + reg_loss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        avg_loss.update_state(total_loss)
        return {"loss": avg_loss.result()}
    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [avg_loss]
    
def custom_loss_with_intermediate(y_true, model_outputs):
    y_pred = model_outputs[0]
    intermediate_output = model_outputs[1]
 # Unpack the outputs
    # Define the loss calculation using both y_pred and intermediate_output
    loss = tf.reduce_mean(tf.square(y_true - y_pred))  # Example: MSE loss for the final output
    # intermediate_loss = some_function_of(intermediate_output)  # Custom logic for intermediate
    return loss  # Combine losses
    
def create_aggregation_model_tr(model, model_2, model_3=None, fusion=None, dense_acivation='relu'):
    embed_size = 512
    num_heads = 8
    hidden_units = [512, 256]
    output_units = 32

    # Initialize the transformer and regression models
    disentangled_transformer = DisentangledTransformer(embed_size, num_heads, hidden_units, output_units)
    regression_model = RegressionMLP(hidden_units=[128, 64], output_units=1)

    class PIDModel(Model):
        def __init__(self,model, model_2, disentangled_transformer, regression_model, optimizer):
            super(PIDModel, self).__init__()
            self.disentangled_transformer = disentangled_transformer
            self.regression_model = regression_model
            self.optimizer =optimizer
            self.model_1c = Model(inputs=model.inputs, outputs=model.layers[-2].output)
            self.model_2c = Model(inputs=model_2.inputs, outputs=model_2.layers[-2].output)

            # Freeze the layers of model_1c and model_2c
            self.model_1c.trainable = False
            self.model_2c.trainable = False

        def call(self, inputs):
            (inputs_1, inputs_2) = inputs
            transformer_outputs = self.disentangled_transformer({'zh': self.model_1c(inputs_1), 'zg': self.model_2c(inputs_2)})
            c = transformer_outputs['c']
            regression_output = self.regression_model(c)
            return regression_output

        def train_step(self, data):
            # Unpack the data. Assuming `data` is a tuple of the form (inputs, targets)
            (inputs_1, inputs_2), targets = data

            with tf.GradientTape() as tape:
                # Forward pass through the connected models
                transformer_output = self.disentangled_transformer({'zh': self.model_1c(inputs_1), 'zg': self.model_2c(inputs_2)})
                regression_output = self.regression_model(transformer_output['c'])

                # Compute the PID loss components
                sh, sg, c = transformer_output['sh'], transformer_output['sg'], transformer_output['c']
                q_c_given_s = transformer_output['q_c_given_s']
                q_sh_given_c = transformer_output['q_sh_given_c']
                q_sg_given_c = transformer_output['q_sg_given_c']
    
                # Calculate the total PID loss
                total_pid_loss = pid_loss(sh, sg, c, q_c_given_s, q_sh_given_c, q_sg_given_c)

                # Calculate the mean squared error for the regression task
                regression_loss = tf.keras.losses.mean_squared_error(targets, regression_output)

                # Total loss is the sum of PID loss and regression loss
                total_loss = 0.1*total_pid_loss + regression_loss

            # Compute gradients and update the weights
            trainable_vars = self.disentangled_transformer.trainable_variables + self.regression_model.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Return a dictionary mapping metrics names to current value
            return {"loss": total_loss, "pid_loss": total_pid_loss, "regression_loss": regression_loss}


    # Compile the model:
    if int(tf.__version__.split('.')[1]) >= 11:
        opt = tf.keras.optimizers.legacy.Adam(lr=0.001)
    else:
        opt = tf.keras.optimizers.Adam(lr=0.001)

    # Metrics
    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        smape
    ]
    pid_model = PIDModel(model, model_2, disentangled_transformer , regression_model, opt)
    pid_model.compile(optimizer='adam')

    # final_model.compile(loss='mse', optimizer=opt, metrics=metrics)

    return pid_model



def create_aggregation_model_orig(model, model_2, model_3=None, fusion=None, dense_acivation='relu'):

    L=128
    D=64

    print(model.summary(), "model1 sum")

    if fusion == 'early' or 'late':
        modelc = Sequential(model.layers[:-1])
        modelc.trainable = False
        model_2c = Sequential(model_2.layers[:-1])
        model_2c.trainable = False
        if model_3:
            model_3 = Sequential(model_3.layers[:-1])
            model_3.trainable = False

    if model_3:
        input3 = model_3.input
        out_3 = model_3(input3)
    # Create a Sequential Model
    input1 = modelc.inputs
    input2 = model_2c.input
    out_1 = modelc(input1)
    out_2 = model_2c(input2)
    out_1c = model(input1)
    out_2c = model_2(input2)

    fusion_features = OuterProductLayer()((out_1,out_2))
    print(fusion_features.shape)


    # Assuming self.fusion_fc is a Keras layer
    x = Dense(256, activation=dense_acivation)(fusion_features)

    x = Dense(128, activation=dense_acivation)(x)
    x = concatenate([x,out_1c,out_2c])
    x = Dense(64, activation=dense_acivation)(x)

    x = Dense(32, activation=dense_acivation)(x)

    x = Dense(16, activation= dense_acivation)(x)
    output_layer = Dense(1)(x)

    #Model Definition
    final_model = Model(inputs=[(input1, input2)], outputs=[output_layer])


    # Compile the model:
    if int(tf.__version__.split('.')[1]) >= 11:
        opt = tf.keras.optimizers.legacy.Adam(lr=0.001)
    else:
        opt = tf.keras.optimizers.Adam(lr=0.001)

    # Metrics
    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        smape
    ]

    final_model.compile(loss='mse', optimizer=opt, metrics=metrics)

    return final_model



def create_aggregation_model_attention(model, model_2, model_3=None, fusion=None, dense_acivation='relu'):

    L=128
    D=64


    if fusion == 'early' or 'late':

        modelc = Sequential(model.layers[:-1])
        modelc.trainable = False
        model_2c = Sequential(model_2.layers[:-1])
        model_2c.trainable = False
        if model_3:

            model_3 = Sequential(model_3.layers[:-1])
            model_3.trainable = False

    if model_3:
        input3 = model_3.input
        out_3 = model_3(input3)

    input1 = modelc.inputs
    input2 = model_2c.input
    out_1 = modelc(input1)
    out_2 = model_2c(input2)
    out_1f = model(input1)
    out_2f = model_2(input2)

    h = tf.keras.layers.Dense(L)(out_1)
    A = Attn_Net_Gated(L, D, dropout=0.2, n_classes=1)(out_1)

    A = tf.nn.softmax(A, axis=1)
    atten_out = A*out_1
    fusion_features = OuterProductLayer()((atten_out,out_2))
    # Assuming self.fusion_fc is a Keras layer
    x = Dense(256, activation=dense_acivation)(fusion_features)
    x = Dense(128, activation=dense_acivation)(x)
    x = concatenate([x,out_1f,out_2f])
    x = Dense(64, activation=dense_acivation)(x)
    x = Dense(32, activation=dense_acivation)(x)
    x = Dense(16, activation=dense_acivation)(x)
    output_layer = Dense(1)(x)
    #Model Definition
    final_model = Model(inputs=[(input1, input2)], outputs=[output_layer])
    # Compile the model:
    opt = keras.optimizers.Adam()

    # Metrics
    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        tf.keras.metrics.MeanAbsoluteError(name='mae'),
        smape
    ]

    final_model.compile(loss='mse', optimizer=opt, metrics=metrics)

    return final_model

def classification_aggregation(model, model_2, model_3=None, fusion=None, dense_acivation='relu'):


    # Create a Sequential Model
    input1 = model.inputs
    input2 = model_2.input

    if fusion == 'early' or 'joint':
        model.layers.pop()
        model_2.layers.pop()
        if model_3:
            model_3.layers.pop()

    if model_3:
        input3 = model_3.input
        out_3 = model_3(input3)

    out_1 = model(input1)
    out_2 = model_2(input2)

    if model_3:
        concat_x = concatenate([out_1, out_2, out_3])
    else:
        concat_x = concatenate([out_1, out_2])

    #Final Layer
    x = Dense(6, activation=dense_acivation)(concat_x)
    output_layer = Dense(3, activation='softmax')(x)

    #Model Definition
    if model_3:
        final_model = Model(inputs=[(input1, input2, input3)], outputs=[output_layer])
    else:
        final_model = Model(inputs=[(input1, input2)], outputs=[output_layer])

    # Compile the model:
    opt = keras.optimizers.Adam()

    # Metrics
    metrics = [
        tf.keras.metrics.AUC(name='auc'), #, multi_label=True, num_labels=3),
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tfa.metrics.F1Score(num_classes=3, threshold=0.5)
        ]

    #loss = tf.keras.losses.SparseCategoricalCrossentropy()

    final_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics)


    return final_model






class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "embed_size must be divisible by num_heads"

        self.wq = layers.Dense(embed_size)
        self.wk = layers.Dense(embed_size)
        self.wv = layers.Dense(embed_size)
        self.dense = layers.Dense(embed_size)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, embed_size)
        k = self.wk(k)  # (batch_size, seq_len, embed_size)
        v = self.wv(v)  # (batch_size, seq_len, embed_size)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, head_dim)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, head_dim)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, head_dim)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        depth = tf.cast(tf.shape(k)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        if mask is not None:
            logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth_v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth_v)
        concat_output = tf.reshape(output, (batch_size, -1, self.embed_size))  # (batch_size, seq_len_q, embed_size)

        output = self.dense(concat_output)  # (batch_size, seq_len_q, embed_size)
        return output

import tensorflow as tf
from tensorflow.keras import Model, layers

# class MLP(Model):
#     def __init__(self, hidden_units, output_units):
#         super(MLP, self).__init__()
#         self.hidden_layers = [layers.Dense(unit, activation='relu', kernel_initializer='he_normal') for unit in hidden_units]
#         self.output_layer = layers.Dense(output_units, activation='relu', kernel_initializer='glorot_uniform')

#     def call(self, inputs):
#         x = inputs
#         for layer in self.hidden_layers:
#             x = layer(x)
#         return self.output_layer(x)
class DisentangledTransformer(tf.keras.Model):
    def __init__(self, embed_size, num_heads, hidden_units, output_units):
        super(DisentangledTransformer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_size)
        # self.mlp_q = MLP(hidden_units, output_units)  # for q(b|a)
        # self.mlp_c = MLP(hidden_units, output_units)  # for q(c|s)
        # self.mlp_a = MLP([240,240],240)
        self.mlp_a = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape = [240]),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(240, activation='relu'),
        ])
        self.mlp_q= tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape = [120]),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(150, activation='relu'),
            tf.keras.layers.Dense(output_units, activation='relu'),
        ])
        self.mlp_c= tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape = [120]),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(150, activation='relu'),
            tf.keras.layers.Dense(output_units, activation='relu'),
        ])
            
        # tf.keras.layers.Dense(100, activation='relu'),
        # tf.keras.layers.Dense(10, activation='softmax')
    def call(self, inputs):
        # Assuming inputs is a dictionary with keys 'zh' and 'zg'
        zh = inputs['zh']  # modality-specific features zh
        zg = inputs['zg']  # modality-specific features zg
        # print(zh.shape,zg.shape,"ZHHH")
        # Concatenate zh and zg for self-attention
        z = tf.concat([zh, zg], axis=1)
        # z = tf.reshape(z, [16, 1, 240]) 
        # print(z,"ZZZZ")
        # Apply multi-head attention
        attn_output = self.mlp_a(z) # self-attention
        # attn_output = z
        print(attn_output,"attn_output")
        # Split features back into zh and zg
        # This would depend on how you want to split them. Here's a simple way:
        split_dim = tf.shape(zh)[1]
        zh_attn, zg_attn = attn_output[:, :split_dim], attn_output[:, split_dim:]

        # Apply the MLP for q(b|a) and q(c|s)
        q_b_given_a_output = self.mlp_q(zh_attn)  # variational approximation q(b|zh)
        # print("+++++++++++========",q_b_given_a_output)
        q_c_given_s_output = self.mlp_c(zg_attn)  # variational approximation q(c|zg)
        
        return {'q_sh_given_c': q_b_given_a_output, 'q_sg_given_c': q_c_given_s_output,'c':z,'sh':zh,'sg':zg,'q_c_given_s':attn_output}

def vclub_loss(q_b_given_a):
    """
    Variational CLUB loss: L_vCLUB(b, a)
    b: Batch of common features C
    a: Batch of input features from which C is generated
    q_b_given_a: Variational approximation network q(b|a)
    """
    # b_given_a = q_b_given_a(a) # MLP to predict b from a
    print(q_b_given_a)
    log_q_b_given_a = tf.math.log(q_b_given_a)

    # Compute the log probabilities for matching pairs
    log_q_b_given_a_matching = tf.linalg.diag_part(log_q_b_given_a)

    # Compute the log probabilities for all possible pairs
    log_q_b_given_a_all = tf.reduce_logsumexp(log_q_b_given_a, axis=1)

    # Variational CLUB loss calculation
    loss = tf.reduce_mean(log_q_b_given_a_matching) - tf.reduce_mean(log_q_b_given_a_all)
    return -loss # Negative because we want to maximize the mutual information

def estimator_loss(b, a, q_b_given_a):
    """
    Estimator loss: L_estimator(b, a)
    """
    pred_b = q_b_given_a# MLP to predict b from a
    log_prob = tf.math.log(tf.clip_by_value(pred_b, 1e-9, 1.0))
    # print(pred_b.shape,log_prob.shape)
    return -tf.reduce_mean(tf.reduce_sum(log_prob, axis=1)) # Negative log likelihood
class RegressionMLP(Model):
    def __init__(self, hidden_units, output_units):
        super(RegressionMLP, self).__init__()
        self.mlp= tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape = [120]),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(150, activation='relu'),
            tf.keras.layers.Dense(1),
        ])

    def call(self, x):
        return self.mlp(x)

# Model instantiation

# Train the PID model using the custom training loop
# pid_model.fit([input_data_1, input_data_2], target_data, epochs=num_epochs)

# Custom training step to handle the complex PID loss and regression loss
# class CustomModel(Model):
#     def train_step(self, data):
#         # Unpack the data
#         inputs, targets = data

#         with tf.GradientTape(persistent=True) as tape:
#             # Forward pass through the transformer
#             transformer_outputs = disentangled_transformer(inputs)

#             # Compute the PID loss components
#             sh, sg, c = transformer_outputs['sh'], transformer_outputs['sg'], transformer_outputs['c']
#             q_c_given_s = transformer_outputs['q_c_given_s']
#             q_sh_given_c = transformer_outputs['q_sh_given_c']
#             q_sg_given_c = transformer_outputs['q_sg_given_c']

#             # Calculate the total PID loss
#             total_pid_loss = pid_loss(sh, sg, c, q_c_given_s, q_sh_given_c, q_sg_given_c)

#             # Forward pass through the regression model
#             regression_output = regression_model(c)
#             # Calculate the mean squared error for the regression task
#             regression_loss = tf.keras.losses.mean_squared_error(targets, regression_output)

#             # Total loss is the sum of PID loss and regression loss
#             total_loss = total_pid_loss + regression_loss

#         # Compute gradients and update the weights
#         trainable_vars = disentangled_transformer.trainable_variables + regression_model.trainable_variables
#         gradients = tape.gradient(total_loss, trainable_vars)
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#         # Return a dictionary mapping metrics names to current value
#         return {"loss": total_loss, "pid_loss": total_pid_loss, "regression_loss": regression_loss}
