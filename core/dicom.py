# mediagent/core/dicom.py
"""
DICOM file parser for MediAgent.
Extracts pixel data + clinical metadata from .dcm files,
converts to base64 PNG for the vision pipeline, and returns
structured metadata to pre-populate the intake form.
"""

import base64
import io
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_dicom(file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Parse a DICOM (.dcm) file.

    Returns:
        (base64_image_string, metadata_dict)
        base64_image_string: "data:image/png;base64,..." ready for vision pipeline
        metadata_dict: extracted clinical metadata for intake pre-population
    """
    try:
        import pydicom
        import numpy as np
        from PIL import Image
    except ImportError as e:
        raise ImportError(f"DICOM support requires pydicom, numpy, Pillow: {e}")

    ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)

    # ── Metadata extraction ───────────────────────────────────────────────────
    metadata: Dict[str, Any] = {}

    _tag_map = {
        "PatientName":       "patient_name",
        "PatientID":         "patient_id",
        "PatientBirthDate":  "birth_date",
        "PatientSex":        "sex",
        "PatientAge":        "age_str",
        "StudyDate":         "study_date",
        "StudyDescription":  "study_description",
        "SeriesDescription": "series_description",
        "Modality":          "modality",
        "InstitutionName":   "institution",
        "Manufacturer":      "manufacturer",
        "ManufacturerModelName": "device_model",
        "KVP":               "kvp",
        "ExposureTime":      "exposure_time_ms",
        "SliceThickness":    "slice_thickness_mm",
        "BodyPartExamined":  "body_part",
        "StudyInstanceUID":  "study_uid",
        "SOPInstanceUID":    "instance_uid",
        "Rows":              "image_rows",
        "Columns":           "image_cols",
        "PixelSpacing":      "pixel_spacing_mm",
    }

    for dicom_tag, key in _tag_map.items():
        try:
            val = getattr(ds, dicom_tag, None)
            if val is not None:
                metadata[key] = str(val)
        except Exception:
            pass

    # Normalise age: DICOM age strings look like "045Y", "006M", "010D"
    age: Optional[int] = None
    age_str = metadata.pop("age_str", None)
    if age_str:
        try:
            if age_str.endswith("Y"):
                age = int(age_str[:-1])
            elif age_str.endswith("M"):
                age = max(0, int(int(age_str[:-1]) / 12))
        except ValueError:
            pass
    if age is not None:
        metadata["age"] = age

    # Normalise sex: DICOM uses M/F/O
    sex = metadata.get("sex", "")
    if sex and sex.upper() in ("M", "F", "O"):
        metadata["sex"] = sex.upper()
    else:
        metadata.pop("sex", None)

    # ── Pixel data → PNG base64 ───────────────────────────────────────────────
    try:
        pixel_array = ds.pixel_array.astype(float)
    except Exception as e:
        raise ValueError(f"Could not read DICOM pixel data: {e}")

    # MONOCHROME1 means bright = low value → invert
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip()
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    # Normalise to 0–255
    p_min, p_max = pixel_array.min(), pixel_array.max()
    if p_max > p_min:
        pixel_array = ((pixel_array - p_min) / (p_max - p_min) * 255).astype("uint8")
    else:
        pixel_array = pixel_array.astype("uint8")

    # Handle grayscale, RGB, multi-frame (take first frame)
    if pixel_array.ndim == 3 and pixel_array.shape[0] > 3:
        pixel_array = pixel_array[0]  # first frame of multi-frame

    if pixel_array.ndim == 2:
        img = Image.fromarray(pixel_array, mode="L").convert("RGB")
    else:
        img = Image.fromarray(pixel_array.astype("uint8"))

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    base64_image = f"data:image/png;base64,{b64}"

    logger.info(
        f"DICOM parsed | modality={metadata.get('modality','?')} "
        f"body_part={metadata.get('body_part','?')} "
        f"size={metadata.get('image_rows','?')}x{metadata.get('image_cols','?')}"
    )
    return base64_image, metadata
