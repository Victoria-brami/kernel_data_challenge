from src.predict import predict

class args:
    def __init__(self):
        self.kernel = 'poly'
        self.sigma = 1
        self.degree = 5
        self.c = 1
        self.classifier_type = 'ova'
        self.feature_extractor = 'hog'
        self.output_file = 'Yte.csv'
        self.train_file = 'None'
        self.modelname = 'svm'
        self.datapath = 'data/'
        self.feature_extractor_cell_size = 4
        self.feature_extractor_cells_per_block = 7
        self.val_split = 0

predict(args())