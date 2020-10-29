from Vision import Sample
from Vision.io_managers import Manager
from Vision.models.classification_model import ClassificationModel
from Vision.pipelines.pipeline import Pipeline


class ClassificationPipe(Pipeline):
    """Pipeline para clasificar imagenes"""
    def __init__(self,sample_manager: Manager, classifier: ClassificationModel):
        super(ClassificationPipe, self).__init__()
        self.sample_manager = sample_manager
        self.classifier = classifier

    def get_sample_manager(self) -> Manager:
        return self.sample_manager

    def predict_sample(self, sample: Sample) -> Sample:
        return self.classifier.predict_sample_append(sample)
