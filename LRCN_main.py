import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Activation, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow.keras.backend as K
from toolbox_glu import moving_average_filter,normalization,segment_signal,transform_data_Re
import argparse
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

K.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = int, default = 0, help = "IR (0), RED(1),IR/RED(2)")
parser.add_argument("--eps", type = int, default = 100, help = "Epochs") 
parser.add_argument("--per", type = int, default = 0, help = "save weight per N epochs")
args = parser.parse_args()

#####  For the dataset
base_directory = 'DatasetRe/'
subjects = os.listdir(base_directory)

# Initialize an empty dictionary
extracted_data = {}

for subject in subjects:
    if subject == '.DS_Store':
        continue
    subject_directory = os.path.join(base_directory, subject)
    
    if os.path.isdir(subject_directory):
        files = os.listdir(subject_directory)
        
        # Initialize 'IR', 'RED', and 'Glucose' to None for each subject
        data_IR = None
        data_RED = None
        numeric_value = None
        
        for file in files:
            if file.endswith('.xlsx'):
                filepath = os.path.join(subject_directory, file)
                
                try:
                    df = pd.read_excel(filepath,engine='openpyxl')
                    
                    if 'IR' in file:
                        data_IR = df['Data']
                    elif 'RED' in file:
                        data_RED = df['Data']
                    elif 'Vital Signs' in file:
                        value = df.iloc[7]
                        glucose = value['Unnamed: 2']
                        numeric_value = int(glucose)
                    
                except Exception as e:
                    # If an error occurs, print a message and continue to the next file
                    print(f"Error loading file {filepath}: {e}")
        
        # Update 'IR', 'RED', and 'Glucose' for the current subject
        extracted_data[subject] = {'IR': data_IR, 'RED': data_RED, 'Glucose': numeric_value}

# the qweight is for constructing the balanced dataset, which is the stepsize of the sliding window
qweight =  np.array([0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
0.,   0.,   0.,   0.,  32.,   0.,  32., 120.,  56.,   0.,  80.,
0.,   8.,  16.,   0.,   0., 128.,  48., 104., 120.,  80.,  48.,
0., 120.,  72.,   0.,  32.,   0.,   0.,  16.,   0.,   0.,  48.,
24.,   0.,  40.,   0.,   0.,   0.,   0.,  16.,   0.,  16.,  16.,
16.,   0.,  16.,  24.,   0.,   0.,  16.,  88.,   0.,  16.,   0.,
8.,   0.,  48.,   0.,  32.,   0.,  32.,   0.,  16.,   0.,  16.,
0.,  64.,   0.,  16.,   0.,  16.,   0.,  24.,   0.,  48.,   0.,
16.,   0.,  16.,  16.,   0.,  48.,   0.,   0.,   0.,  16.,  32.,
16.,   0.,  72.,  32.,   0.,  32.,  16.,   0.,   0.,  16.,   0.,
0.,   0.,  32.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
16.,   0.,  32.,   0.,   0.,   0.,   0.,  16.,   0.,   0.,   0.,
0.,   0.,   0.,  24.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
0.,   0.,   0.,   0.,   0.,  32.,   0.,   0.,   0.,   0.,   0.
])


IR_matrix, RED_matrix, Glucose_IR, Glucose_RED = transform_data_Re(extracted_data,qweight) 
kf = KFold(n_splits=5, shuffle=True, random_state=42)

if args.dataset == 0:
    X = IR_matrix
    y = Glucose_IR
elif args.dataset == 1:
    X = RED_matrix
    y = Glucose_RED
elif args.dataset == 2:
    X = np.concatenate(
        (IR_matrix, RED_matrix), axis=0
    )
    y = np.append(Glucose_IR, Glucose_RED)


# Define the Conv1D_Block function
def Conv1D_Block(input_tensor, filters, dropout_rate=0.1):
    x = Conv1D(filters, kernel_size=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    return x

input_layer = Input(shape=(128, 1))

x = Conv1D_Block(input_layer, 64)
x = MaxPooling1D(pool_size=2, strides=2)(x)

x = Conv1D_Block(x, 128)
x = MaxPooling1D(pool_size=2, strides=2)(x)

x = Conv1D_Block(x, 256)
x = Conv1D_Block(x, 256)
x = MaxPooling1D(pool_size=2, strides=2)(x)

x = LSTM(256,dropout=0.2)(x)

output_layer = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
adam = tf.keras.optimizers.Adam(lr=0.0001)
save_path = "weight/" + "beta/No" + str(args.dataset) +"glucose" + "-{epoch:02d}-{val_accuracy:.4f}.h5"
if args.per!= 0:
    checkpoints = tf.keras.callbacks.ModelCheckpoint(save_path, save_weights_only=True, period=args.per)
    exp_callback = [tf.keras.callbacks.EarlyStopping(patience=500), checkpoints]
else:
    exp_callback = [tf.keras.callbacks.EarlyStopping(patience=500)]


model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])

model.summary()

epochs = args.eps
batch_size = 32
#
training_loss_minmax = []
validation_loss_minmax = []
training_loss_MAE = []
validation_loss_MAE = []
incc = 0

for train_index, val_index in kf.split(X):
    incc = incc+1
    x_train, x_test = X[train_index].reshape((X[train_index].shape[0], X[train_index].shape[1], 1)), X[val_index].reshape((X[val_index].shape[0], X[val_index].shape[1], 1))
    y_train, y_test = y[train_index], y[val_index]

    # Min-Max Scaling
    y_train_scaled = (y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train))
    y_test_scaled = (y_test - np.min(y_train)) / (np.max(y_train) - np.min(y_train))
    tude = np.max(y_train) - np.min(y_train)

    exp_history = model.fit(x_train, y_train_scaled, batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(x_test,y_test_scaled), callbacks=exp_callback)
    

    # Evaluate the model using scaled labels
    score_train_scaled = model.evaluate(x_train, y_train_scaled, verbose=0)
    score_test_scaled = model.evaluate(x_test, y_test_scaled, verbose=0)

    original_train_loss = score_train_scaled[0] * (tude**2)
    original_test_loss = score_test_scaled[0] * (tude**2)

    original_train_mae = score_train_scaled[1] * tude
    original_test_mae = score_test_scaled[1] * tude

    training_loss_minmax.append(original_train_loss)
    validation_loss_minmax.append(original_test_loss)
    training_loss_MAE.append(original_train_mae)
    validation_loss_MAE.append(original_test_mae)

    # Convert the loss values to the original scale
    original_train_curve = [loss * (tude**2) for loss in exp_history.history['loss']]
    original_val_curve = [loss * (tude**2) for loss in exp_history.history['val_loss']]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(original_train_curve)
    plt.plot(original_val_curve)
    plt.title('Model loss in Original Scale')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f"img/loss_minmax_original_scale {incc}.png")  


# Average Losses across Folds
avg_train_loss_minmax = np.mean(training_loss_minmax, axis=0)
avg_val_loss_minmax = np.mean(validation_loss_minmax, axis=0)
avg_train_loss_MAE = np.mean(training_loss_MAE, axis=0)
avg_val_loss_MAE = np.mean(validation_loss_MAE, axis=0)
