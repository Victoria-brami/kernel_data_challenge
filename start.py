from src.predict import predict

class args:
    def __init__(self):
        self.kernel = 'rbf'
        self.sigma = 1
        self.degree = 5
        self.c = 1
        self.classifier_type = 'ovo'
        self.feature_extractor = 'hog'
        self.output_file = 'Yte.csv'
        self.train_file = 'None'
        self.modelname = 'svm'
        self.datapath = 'data/'
        self.feature_extractor_cell_size = 8
        self.feature_extractor_cells_per_block = 3
        self.val_split = 0

predict(args())