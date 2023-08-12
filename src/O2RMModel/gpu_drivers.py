import tensorflow

physical_devices = tensorflow.config.list_physical_devices('GPU')

for p in physical_devices:
    print(p)

