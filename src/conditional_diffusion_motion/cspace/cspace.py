import numpy as np
import pinocchio as pin
import time

class CSpace:
    def __init__(self, rmodel: pin.Model, cmodel: pin.GeometryModel):
        """
        C-space evaluator using Pinocchio for collision checking.

        Args:
            rmodel (pin.Model): Robot model.
            cmodel (pin.GeometryModel): Geometry model with defined collision pairs.
        """
        self._rmodel = rmodel
        self._cmodel = cmodel
        self._rdata = rmodel.createData()
        self._cdata = cmodel.createData()

        self._configurations = []
        self._validity_mask = []

    def load_configurations(self, file_path: str):
        """Load a list of robot configurations from a .npy file."""
        self._configurations = np.load(file_path, allow_pickle=True).tolist()

    def set_configurations(self, configs: list[np.ndarray]):
        """Directly set configurations from a list."""
        self._configurations = configs

    def _is_in_cspace(self, configuration) -> bool:
        if not isinstance(configuration, np.ndarray):
            try: 
                configuration = np.array(configuration)
            except Exception as e:
                raise ValueError(f"Invalid configuration format: {e}")
        pin.forwardKinematics(self._rmodel, self._rdata, configuration)
        pin.updateGeometryPlacements(self._rmodel, self._rdata, self._cmodel, self._cdata)
        return not pin.computeCollisions(self._cmodel, self._cdata)

    def evaluate(self, verbose=False):
        """
        Evaluate and cache the validity of all loaded configurations.
        """
        self._validity_mask = []
        for idx, config in enumerate(self._configurations):
            is_valid = self._is_in_cspace(config)
            self._validity_mask.append(is_valid)
            if verbose and not is_valid:
                print(f"[CSpace] Configuration {idx} is in collision.")

    def is_valid(self, configuration: np.ndarray) -> bool:
        """Check if a single configuration is valid (i.e., not in collision)."""
        return self._is_in_cspace(configuration)

    @property
    def validity_mask(self) -> np.ndarray:
        """Returns a boolean array indicating C-space validity."""
        return np.array(self._validity_mask, dtype=bool)

    @property
    def valid_configurations(self) -> list[np.ndarray]:
        """Returns only valid configurations."""
        return [cfg for cfg, valid in zip(self._configurations, self._validity_mask) if valid]

    @property
    def invalid_configurations(self) -> list[np.ndarray]:
        """Returns only invalid configurations."""
        return [cfg for cfg, valid in zip(self._configurations, self._validity_mask) if not valid]
