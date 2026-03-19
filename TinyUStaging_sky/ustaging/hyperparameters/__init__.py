import ustaging
from mpunet.hyperparameters import YAMLHParams as _YAMLHParams


class YAMLHParams(_YAMLHParams):
    """
    Wraps the YAMLHParams class from MultiPlanarUNet, passing 'ustaging' as the
    package for correct version controlling.
    """
    def __init__(self, *args, **kwargs):
        kwargs["package"] = ustaging.__name__
        super(YAMLHParams, self).__init__(
            *args, **kwargs
        )
