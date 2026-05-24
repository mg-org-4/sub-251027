from .src.spectrum_node import SpectrumSDXL
from .src.calibrated_spectrum_node_legacy import SpectrumSDXLCalibrated

NODE_CLASS_MAPPINGS = {
    "SpectrumSDXL": SpectrumSDXL,
    "CalibratedSpectrumSDXL": SpectrumSDXLCalibrated,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectrumSDXL": "Spectrum Adaptive Forecaster (SDXL)",
    "CalibratedSpectrumSDXL": "Calibrated Spectrum Adaptive Forecaster (SDXL) [LEGACY]",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]