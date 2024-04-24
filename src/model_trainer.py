from tensorflow.keras.optimizers import Adam
from src.model_builder import ModelBuilder

class ModelTrainer:
    def __init__(self, data_path, batch_size, batch_num, epoch):
        self.data_path = data_path
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.epoch = epoch

    def load_data(self):
        # Load data from CSV
        # Split into training and testing sets
        return X_train, X_test, Y_train, Y_test

    def batch_generator(self, X_data, Y_data, train_flag):
        # Generate batches of images and corresponding labels
        # Apply preprocessing if train_flag is True
        yield image_container, steer_container

    def train_model(self):
        X_train, X_test, Y_train, Y_test = self.load_data()
        input_shape = (66, 200, 3)  # Example input shape
        model_builder = ModelBuilder(input_shape)
        model = model_builder.build_model2()  # Example model selection
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])
        model.fit(
            self.batch_generator(X_train, Y_train, True),
            steps_per_epoch=self.batch_num,
            epochs=self.epoch,
            verbose=1,
            validation_data=self.batch_generator(X_test, Y_test, False),
            validation_steps=1,
            max_queue_size=1,
            callbacks=[checkpoint, stopping, tensor_board]
        )
        model.save('xinglina_lake_model2.h5')
