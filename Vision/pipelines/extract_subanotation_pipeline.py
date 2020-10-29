from Vision import Sample
from Vision.io_managers import Manager
from Vision.pipelines.pipeline import Pipeline


class ExtractSubAnotationPipe(Pipeline):
    """Extrae una un subset de anotaciones de las muestras"""

    def get_sample_manager(self) -> Manager:
        return self.man_in

    def predict_sample(self, sample: Sample) -> Sample:
        r_pattern = self.r_pattern
        min_annotations = self.min_annotations
        if len(sample.filter_annotations(r_pattern)) >= min_annotations:
            if self.zoom:
                new_sample = sample.zoom_to_annotations(cls_filt=r_pattern)
                return new_sample
            else:
                new_sample = sample.substitute_annotations(
                    sample.filter_annotations(r_pattern=r_pattern)
                )
                return new_sample

    def __init__(self,
                 manager_in: Manager,
                 manager_out: Manager = None,
                 r_pattern: str = r'.*',
                 zoom=True,
                 min_annotations=0
                 ):
        """
        :param manager_in:
        :type manager_in:
        :param manager_out:
        :type manager_out:
        :param r_pattern:
        :type r_pattern:
        :param zoom:
        :type zoom:
        :param min_annotations:
        :type min_annotations:
        """
        super(ExtractSubAnotationPipe, self).__init__()
        self.min_annotations = min_annotations
        self.man_in = manager_in
        if manager_out is None:
            self.man_out = manager_in
        else:
            self.man_out = manager_out
        self.r_pattern = r_pattern
        self.zoom = zoom

    def __call__(self, batch_size=32, *args, **kwargs):
        r_pattern = self.r_pattern
        man_in = self.man_in
        man_out = self.man_out
        min_annotations = self.min_annotations
        counter = 0
        for samples in man_in.sample_generator(batch_size):
            for sample in samples:
                if len(sample.filter_annotations(r_pattern)) >= min_annotations:
                    if self.zoom:
                        new_sample = sample.zoom_to_annotations(cls_filt=r_pattern)
                    else:
                        new_sample = sample.substitute_annotations(
                            sample.filter_annotations(r_pattern=r_pattern)
                        )
                    man_out.write_sample(new_sample, True)
                    counter += 1
                    print(f"{counter} samples extracted!")
