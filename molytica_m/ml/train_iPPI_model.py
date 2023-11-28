from tensorflow.keras.callbacks import ModelCheckpoint
from molytica_m.ml.iPPI_model import create_iPPI_model
from molytica_m.data_tools import dataset_tools

model = create_iPPI_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

disjoint_loader_train = dataset_tools.get_disjoint_loader("data/iPPI_data/train/", batch_size=50, epochs=100000)
disjoint_loader_val = dataset_tools.get_disjoint_loader("data/iPPI_data/val/", batch_size=50)

checkpoint = ModelCheckpoint("molytica_m/ml/iPPI_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


print(disjoint_loader_train.load())

model.fit(
        disjoint_loader_train.load(),
        validation_data=disjoint_loader_val.load(),
        validation_steps=disjoint_loader_val.steps_per_epoch,
        steps_per_epoch=disjoint_loader_train.steps_per_epoch,
        epochs=10000000,
        callbacks=[checkpoint] 
    )
