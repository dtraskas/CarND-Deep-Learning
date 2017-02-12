#
# Deep Learning Autonomous Car
# Configurator module with all the parameters of the model
#
# Dimitrios Traskas
#
#

class Configurator:
    
    def __init__(self, data_path, image_shape=(160, 320, 3), reduced_shape=(80, 160, 3), batch_size=128, split_size=0.2):
        self.data_path = data_path        
        self.image_shape = image_shape
        self.reduced_shape = reduced_shape
        self.batch_size = batch_size
        self.split_size = split_size