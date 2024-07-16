from model import *
import pytz
from tensorflow.keras.models import load_model
from datetime import datetime



def lr_schedule(epoch, lr):
    """
    Reduce the learning rate every 7 epochs by a factor of 0.5.
    """
    if (epoch + 1) % 10 == 0:
        return lr * 0.8
    return lr

def train_model(model, checkpoint_dir_path , train_dataset, val_dataset, optimizer,lr_schedule=lr_schedule, use_early_stoping=False ,epochs=100, early_stopping_patience=10):
    
    if isinstance(model, str):
        model = load_model(model, custom_objects={'CTCLayer': CTCLayer}, compile=False)
    

    # Define the ModelCheckpoint callback with the epoch number, train loss, val loss, and Egypt time
    checkpoint = CustomModelCheckpoint(
        filepath=checkpoint_dir_path+'/captcha_CF_TF_model_epoch_{epoch}_train_loss_{train_loss:.4f}_val_loss_{val_loss:.4f}_{egypt_time}.h5',
        monitor='val_loss',
        mode='min'
    )

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )

    # opt = tf.keras.optimizers.Adam(learning_rate=0.00008)
    model.compile(optimizer=optimizer)
    
    if use_early_stoping:
        callbacks=[checkpoint, lr_scheduler, early_stopping]
    else:
        callbacks=[checkpoint, lr_scheduler]
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )

    return history, model








def get_egypt_time():
    tz = pytz.timezone('Africa/Cairo')
    return datetime.now(tz).strftime("%Y%m%d-%H%M%S")

class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', mode='min'):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        if mode == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if self.mode == 'min' and current < self.best:
            self.best = current
            self.save_model(epoch, train_loss, val_loss)
        elif self.mode == 'max' and current > self.best:
            self.best = current
            self.save_model(epoch, train_loss, val_loss)
        else:
            self.save_model(epoch, train_loss, val_loss)  # Save every epoch

    def save_model(self, epoch, train_loss, val_loss):
        file_path = self.filepath.format(epoch=epoch+1, train_loss=train_loss, val_loss=val_loss, egypt_time=get_egypt_time())
        self.model.save(file_path)
        print(f'Model saved to {file_path}')