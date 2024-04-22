from src.components.model_builder import ModelBuilder

if __name__ == '__main__':
    # Set input shape
    input_shape = (66, 200, 3)

    # Instantiate ModelBuilder
    model_builder = ModelBuilder(input_shape)

    # Build models
    model1 = model_builder.build_model(1)
    model2 = model_builder.build_model(2)
    model3 = model_builder.build_model(3)

    # Print model summaries
    print("Model 1 Summary:")
    model1.summary()

    print("\nModel 2 Summary:")
    model2.summary()

    print("\nModel 3 Summary:")
    model3.summary()
