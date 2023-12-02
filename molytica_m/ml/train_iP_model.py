from tensorflow.keras.callbacks import ModelCheckpoint
from molytica_m.ml.iP_model import create_iP_model
from molytica_m.data_tools import dataset_tools

model = create_iP_model()
model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mean_squared_error'])

disjoint_loader_train = dataset_tools.get_disjoint_loader("data/iP_data/train/", batch_size=30, epochs=100000)
disjoint_loader_val = dataset_tools.get_disjoint_loader("data/iP_data/val/", batch_size=30)

checkpoint = ModelCheckpoint("molytica_m/ml/iP_model.h5", monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')


print(disjoint_loader_train.load())

model.fit(
        disjoint_loader_train.load(),
        validation_data=disjoint_loader_val.load(),
        validation_steps=disjoint_loader_val.steps_per_epoch,
        steps_per_epoch=disjoint_loader_train.steps_per_epoch,
        epochs=10000000,
        callbacks=[checkpoint] 
    )
