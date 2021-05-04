from interface import DDPGInterface

from keras import layers
import keras

import numpy as np
import os



def build_task_predictor(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

img_shape = (96, 96, 1)

num_train_samples = 100
num_val_samples = 50
num_holdout_samples = 50

x_train = np.random.rand(num_train_samples, img_shape[0], img_shape[1], img_shape[2])
y_train = np.random.randint(low=0, high=2, size=(num_train_samples, 1))

x_val = np.random.rand(num_val_samples, img_shape[0], img_shape[1], img_shape[2])
y_val = np.random.randint(low=0, high=2, size=(num_val_samples, 1))

x_holdout = np.random.rand(num_holdout_samples, img_shape[0], img_shape[1], img_shape[2])
y_holdout = np.random.randint(low=0, high=2, size=(num_holdout_samples, 1))

task_predictor = build_task_predictor(img_shape)
task_predictor.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # speciffy the loss and metric used to train target net and controller respectively


def build_actor_critic(img_shape, action_shape=(1,)):
    
    n_actions = action_shape[0]
    
    act_in = layers.Input((1,) + img_shape)
    act_in_reshape = layers.Reshape((img_shape))(act_in)
    act_x = layers.Conv2D(32, (3,3), activation='relu')(act_in_reshape)
    act_x = layers.MaxPool2D((2,2))(act_x)
    act_x = layers.Conv2D(64, (3,3), activation='relu')(act_x)
    act_x = layers.MaxPool2D((2,2))(act_x)
    act_x = layers.Conv2D(64, (3,3), activation='relu')(act_x)
    act_x = layers.Flatten()(act_x)
    act_x = layers.Dense(64, activation='relu')(act_x)
    act_x = layers.Dense(32, activation='relu')(act_x)
    act_x = layers.Dense(16, activation='relu')(act_x)
    act_out = layers.Dense(n_actions, activation='sigmoid')(act_x)
    actor = keras.Model(inputs=act_in, outputs=act_out)
    
    action_input = layers.Input(shape=(n_actions,), name='action_input')
    observation_input = layers.Input((1,) + img_shape, name='observation_input')
    observation_input_reshape = layers.Reshape((img_shape))(observation_input)
    observation_x = layers.Conv2D(32, (3,3), activation='relu')(observation_input_reshape)
    observation_x = layers.MaxPool2D((2,2))(observation_x)
    observation_x = layers.Conv2D(64, (3,3), activation='relu')(observation_x)
    observation_x = layers.MaxPool2D((2,2))(observation_x)
    observation_x = layers.Conv2D(64, (3,3), activation='relu')(observation_x)
    flattened_observation = layers.Flatten()(observation_x)
    x = layers.Concatenate()([action_input, flattened_observation])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(1)(x)
    critic = keras.Model(inputs=[action_input, observation_input], outputs=x)
    return actor, critic, action_input

actor, critic, action_input = build_actor_critic(img_shape)

interface = DDPGInterface(x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape,
                          custom_controller=True, actor=actor, critic=critic, action_input=action_input,
                          modify_env_params=True, modified_env_params_list=[60, 30])


interface.train(6)

save_dir = '/home/s-sd/Desktop/task_amenability_repo/temp'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

save_path = os.path.join(save_dir, 'pneumoniamnist_experiment_train_session')

controller_weights_save_path = save_path + 'abc'
task_predictor_save_path = save_path + 'def'
interface.save(controller_weights_save_path=controller_weights_save_path,
               task_predictor_save_path=task_predictor_save_path)

del interface
del actor 
del critic
del action_input

actor, critic, action_input = build_actor_critic(img_shape)

interface = DDPGInterface(x_train, y_train, x_val, y_val, x_holdout, y_holdout, task_predictor, img_shape, 
                          load_models=True, controller_weights_save_path=controller_weights_save_path, task_predictor_save_path=task_predictor_save_path,
                          custom_controller=True, actor=actor, critic=critic, action_input=action_input,
                          modify_env_params=True, modified_env_params_list=[60, 30])

holdout_controller_preds = interface.get_controller_preds_on_holdout()

def reject_lowest_controller_valued_samples(rejection_ratio, holdout_controller_preds, x_holdout, y_holdout):
    sorted_inds = np.argsort(holdout_controller_preds)
    num_rejected = int(rejection_ratio * len(sorted_inds))
    selected_x_holdout, selected_y_holdout = x_holdout[sorted_inds[num_rejected:], :, :, :], y_holdout[sorted_inds[num_rejected:]]
    return selected_x_holdout, selected_y_holdout

def compute_mean_performance(x, y, interface):
    mean_performance_metric = interface.task_predictor.evaluate(x, y)
    return mean_performance_metric[-1]

selected_x_holdout, selected_y_holdout = reject_lowest_controller_valued_samples(0.9, holdout_controller_preds, x_holdout, y_holdout)
compute_mean_performance(selected_x_holdout, selected_y_holdout, interface)

