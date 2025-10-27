# train.py
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Dropout,
                                     Bidirectional, LSTM, Dense, GlobalAveragePooling1D, Attention,
                                     Add, Multiply, Reshape)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def attention_block(x, ratio=8):
    channel = int(x.shape[-1])
    shared_dense_one = Dense(max(1, channel // ratio), activation='relu')
    shared_dense_two = Dense(channel, activation='sigmoid')

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    avg_pool = Reshape((1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = Reshape((1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)
    return Multiply()([x, cbam_feature])

def build_cnn_attention(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(128, 7, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = attention_block(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = attention_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_highacc_cnn_bilstm_attention(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(128, 7, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    attn = Attention()([x, x])
    x = GlobalAveragePooling1D()(attn)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=3e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train(file_path, mode="strong_weak", n_splits=5, epochs=5, batch_size=32):
    data = pd.read_csv(file_path)
    if 'class' in data.columns:
        y = data['class'].values
        X = data.drop(columns=['class']).values
    else:
        y = data.iloc[:, 0].values
        X = data.iloc[:, 1:].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if mode == "promoter":
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_scaled, y)
        model_builder = build_cnn_attention
    else:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_scaled, y)
        model_builder = build_highacc_cnn_bilstm_attention

    X_res = X_res.reshape(X_res.shape[0], X_res.shape[1], 1)
    input_shape = (X_res.shape[1], 1)
    joblib.dump(scaler, "scaler.joblib")

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X_res, y_res):
        X_train, X_val = X_res[train_idx], X_res[val_idx]
        y_train, y_val = y_res[train_idx], y_res[val_idx]
        model = model_builder(input_shape)
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch_size, callbacks=[es, lr], verbose=2)
        model.save("model_final.h5")
        break

    print("✅ Training complete. Saved model_final.h5 and scaler.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to CSV (e.g., 222_vecs.csv or SW_222_vecs.csv)")
    parser.add_argument("--mode", choices=["promoter", "strong_weak"], default="strong_weak")
    args = parser.parse_args()
    train(args.file, args.mode)
