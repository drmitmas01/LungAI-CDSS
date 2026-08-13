from datetime import datetime
from pathlib import Path
from typing import Any

import configparser
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from skimage import measure


# =========================================================
# COMPATIBILITY FIXES FOR OLDER PYLIDC
# =========================================================
# Python 3.12+ removed SafeConfigParser.
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

# Recent NumPy versions removed these aliases, but some older
# pylidc code still expects them.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

# Import pylidc only after applying compatibility fixes.
import pylidc as pl
from pylidc.utils import consensus


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="LungAI-CDSS",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# FILE PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = PROJECT_DIR / "lidc_model_dataset_binary.csv"
PERFORMANCE_PATH = PROJECT_DIR / "model_performance_results.csv"

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
IMPUTER_PATH = BASE_DIR / "models" / "imputer.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"
STYLE_PATH = BASE_DIR / "style.css"


# =========================================================
# LOAD CSS
# =========================================================
def load_css() -> None:
    """Load custom dashboard styling from style.css."""
    if STYLE_PATH.exists():
        css = STYLE_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css()


# =========================================================
# LOAD SAVED MODEL ASSETS AND DATA
# =========================================================
@st.cache_resource
def load_model_assets() -> tuple[Any, Any, list[str]]:
    """Load the trained model, imputer and ordered feature-name list."""
    loaded_model = joblib.load(MODEL_PATH)
    loaded_imputer = joblib.load(IMPUTER_PATH)
    loaded_features = joblib.load(FEATURE_PATH)

    return loaded_model, loaded_imputer, list(loaded_features)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the prepared LIDC-IDRI modelling dataset."""
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_performance_results() -> pd.DataFrame:
    """Load the saved model-comparison results."""
    return pd.read_csv(PERFORMANCE_PATH)


# =========================================================
# CT IMAGE AND ANNOTATION UTILITIES
# =========================================================
@st.cache_data(show_spinner=False)
def load_ct_case(
    patient_id: str,
    nodule_number: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load the CT volume for one LIDC-IDRI patient and locate the selected
    nodule using the mean centroid of the available radiologist annotations.

    Returns
    -------
    volume:
        CT image volume with shape (rows, columns, slices).
    centroid:
        Approximate nodule centre as [row, column, slice].
    annotation_count:
        Number of radiologist annotations in the selected cluster.
    """

    scan = (
        pl.query(pl.Scan)
        .filter(pl.Scan.patient_id == patient_id)
        .first()
    )

    if scan is None:
        raise ValueError(
            f"No pylidc scan was found for patient {patient_id}."
        )

    volume = scan.to_volume(verbose=False)
    clusters = scan.cluster_annotations(verbose=False)

    cluster_index = int(nodule_number) - 1

    if cluster_index < 0 or cluster_index >= len(clusters):
        raise IndexError(
            f"Nodule {nodule_number} is unavailable for patient "
            f"{patient_id}. This scan contains {len(clusters)} "
            "clustered nodules."
        )

    selected_cluster = clusters[cluster_index]

    if len(selected_cluster) == 0:
        raise ValueError(
            "The selected nodule cluster does not contain annotations."
        )

    centroids = np.array(
        [annotation.centroid for annotation in selected_cluster],
        dtype=float,
    )

    mean_centroid = centroids.mean(axis=0)

    return volume, mean_centroid, len(selected_cluster)


@st.cache_data(show_spinner=False)
def load_nodule_3d_mesh(
    patient_id: str,
    nodule_number: int,
    consensus_level: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], int, int, float]:
    """
    Create a triangular 3D mesh from the consensus radiologist
    segmentation of the selected LIDC-IDRI nodule.

    Returns
    -------
    vertices:
        Mesh vertex coordinates in millimetres.
    faces:
        Triangular face indices.
    mask_shape:
        Consensus-mask dimensions.
    annotation_count:
        Number of radiologist annotations in the cluster.
    individual_mask_count:
        Number of individual annotation masks combined.
    approximate_volume_mm3:
        Approximate consensus-mask volume in cubic millimetres.
    """

    scan = (
        pl.query(pl.Scan)
        .filter(pl.Scan.patient_id == patient_id)
        .first()
    )

    if scan is None:
        raise ValueError(
            f"No scan was found for patient {patient_id}."
        )

    clusters = scan.cluster_annotations(verbose=False)
    cluster_index = int(nodule_number) - 1

    if cluster_index < 0 or cluster_index >= len(clusters):
        raise IndexError(
            f"Nodule {nodule_number} could not be located. "
            f"This patient has {len(clusters)} clustered nodules."
        )

    selected_cluster = clusters[cluster_index]

    if len(selected_cluster) == 0:
        raise ValueError(
            "The selected nodule cluster does not contain annotations."
        )

    consensus_mask, _bounding_box, individual_masks = consensus(
        selected_cluster,
        clevel=float(consensus_level),
        pad=[
            (3, 3),
            (3, 3),
            (2, 2),
        ],
        ret_masks=True,
    )

    if consensus_mask is None or not np.any(consensus_mask):
        raise ValueError(
            "The selected consensus level produced an empty mask. "
            "Try reducing the consensus level."
        )

    pixel_spacing = getattr(scan, "pixel_spacing", None)

    if pixel_spacing is None:
        pixel_spacing = 1.0

    pixel_spacing = float(pixel_spacing)

    slice_spacing = getattr(scan, "slice_spacing", None)

    if slice_spacing is None:
        slice_spacing = getattr(scan, "slice_thickness", None)

    if slice_spacing is None:
        slice_spacing = 1.0

    slice_spacing = float(slice_spacing)

    spacing = (
        pixel_spacing,
        pixel_spacing,
        slice_spacing,
    )

    vertices, faces, _normals, _values = measure.marching_cubes(
        consensus_mask.astype(np.uint8),
        level=0.5,
        spacing=spacing,
    )

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(
            "A valid 3D surface could not be extracted from the "
            "consensus mask."
        )

    voxel_volume_mm3 = (
        pixel_spacing
        * pixel_spacing
        * slice_spacing
    )

    approximate_volume_mm3 = float(
        np.count_nonzero(consensus_mask)
        * voxel_volume_mm3
    )

    return (
        vertices,
        faces,
        consensus_mask.shape,
        len(selected_cluster),
        len(individual_masks),
        approximate_volume_mm3,
    )


def apply_ct_window(
    image: np.ndarray,
    centre: float,
    width: float,
) -> np.ndarray:
    """Apply a CT display window and normalise the output to 0–1."""

    lower = centre - width / 2
    upper = centre + width / 2

    windowed = np.clip(image, lower, upper)
    denominator = upper - lower

    if denominator <= 0:
        return np.zeros_like(windowed, dtype=float)

    return (windowed - lower) / denominator


# =========================================================
# LOAD APPLICATION DATA
# =========================================================
try:
    model, imputer, feature_names = load_model_assets()
    df = load_dataset()

except FileNotFoundError as exc:
    st.error(f"Required file not found: {exc}")
    st.stop()

except Exception as exc:
    st.error(f"Dashboard could not be loaded: {exc}")
    st.stop()


try:
    performance_df = load_performance_results()

except Exception:
    performance_df = pd.DataFrame()


# =========================================================
# HEADER
# =========================================================
header_html = (
    '<div class="main-header">'
        '<div>'
            '<h1>LungAI-CDSS</h1>'
            '<p class="header-subtitle">'
                'Explainable AI Clinical Decision Support System<br>'
                'for CT-Based Lung Nodule Malignancy Risk Prediction'
            '</p>'
            '<div class="system-status">'
                '<span class="status-dot"></span>'
                'System Status: Ready'
            '</div>'
        '</div>'
        '<div class="prototype-badge">'
            'LungAI-CDSS<br>'
            '<span>Research Prototype</span>'
        '</div>'
    '</div>'
)

st.html(header_html)

st.warning(
    "This prototype is for academic research only. It is not clinically "
    "validated and must not be used for direct patient diagnosis or "
    "treatment decisions."
)


# =========================================================
# SIDEBAR CASE SELECTION
# =========================================================
st.sidebar.markdown("## Case Selection")

patient_ids = sorted(
    df["patient_id"]
    .dropna()
    .astype(str)
    .unique()
)

selected_patient = st.sidebar.selectbox(
    "Patient ID",
    patient_ids,
)

patient_df = df[
    df["patient_id"].astype(str) == selected_patient
].copy()

nodule_numbers = sorted(
    patient_df["nodule_number"]
    .dropna()
    .astype(int)
    .unique()
)

selected_nodule = st.sidebar.selectbox(
    "Nodule Number",
    nodule_numbers,
)

selected_rows = patient_df[
    patient_df["nodule_number"].astype(int) == selected_nodule
]

if selected_rows.empty:
    st.error(
        "The selected patient and nodule record could not be found."
    )
    st.stop()

selected_row = selected_rows.iloc[0]


# =========================================================
# PREPARE FEATURES AND MAKE PREDICTION
# =========================================================
missing_features = [
    feature
    for feature in feature_names
    if feature not in selected_row.index
]

if missing_features:
    st.error(
        "The selected dataset record is missing required model "
        f"features: {missing_features}"
    )
    st.stop()


case_features = pd.DataFrame(
    [selected_row[feature_names].to_dict()],
    columns=feature_names,
)

case_imputed = imputer.transform(case_features)

prediction = int(
    model.predict(case_imputed)[0]
)

probabilities = model.predict_proba(
    case_imputed
)[0]

low_probability = float(
    probabilities[0]
)

suspicious_probability = float(
    probabilities[1]
)

predicted_label = (
    "Suspicious"
    if prediction == 1
    else "Low Risk"
)

confidence = max(
    low_probability,
    suspicious_probability,
)

actual_group = str(
    selected_row.get(
        "risk_group",
        "Not available",
    )
)

mean_malignancy = selected_row.get(
    "mean_malignancy_score",
    None,
)


# =========================================================
# CLINICAL INTERPRETATION AND RECOMMENDATION
# =========================================================
if prediction == 1:
    interpretation = (
        "The model classifies this nodule as suspicious risk. "
        "The available CT-derived characteristics are more consistent "
        "with the suspicious-risk group."
    )

    recommendation = (
        "Suggested decision-support action: consider further "
        "radiological review, multidisciplinary assessment or "
        "additional investigation where clinically appropriate."
    )

else:
    interpretation = (
        "The model classifies this nodule as low risk based on the "
        "available CT-derived characteristics."
    )

    recommendation = (
        "Suggested decision-support action: consider routine "
        "surveillance or follow-up in accordance with applicable "
        "clinical guidance and professional judgement."
    )


# =========================================================
# SUMMARY CARDS
# =========================================================
risk_class = (
    "risk-suspicious"
    if prediction == 1
    else "risk-low"
)

risk_icon = (
    "⚠️"
    if prediction == 1
    else "✅"
)

confidence_text = (
    "Very high confidence"
    if confidence >= 0.90
    else "Moderate confidence"
    if confidence >= 0.70
    else "Borderline confidence"
)

summary_cols = st.columns(4)

with summary_cols[0]:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="card-label">Patient ID</div>
            <div class="card-value">{selected_patient}</div>
            <div class="card-caption">Selected imaging case</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_cols[1]:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="card-label">Nodule Number</div>
            <div class="card-value">{selected_nodule}</div>
            <div class="card-caption">Annotated lung nodule</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_cols[2]:
    st.markdown(
        f"""
        <div class="summary-card {risk_class}">
            <div class="card-label">Predicted Risk</div>
            <div class="card-value">{risk_icon} {predicted_label}</div>
            <div class="card-caption">AI-generated classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_cols[3]:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="card-label">Model Confidence</div>
            <div class="card-value">{confidence:.1%}</div>
            <div class="card-caption">{confidence_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# WORKFLOW INDICATOR
# =========================================================
workflow_html = (
    '<div class="workflow-container">'
        '<div class="workflow-step completed">'
            '<span>1</span>'
            'Patient Selected'
        '</div>'
        '<div class="workflow-arrow">&rarr;</div>'
        '<div class="workflow-step completed">'
            '<span>2</span>'
            'AI Risk Prediction'
        '</div>'
        '<div class="workflow-arrow">&rarr;</div>'
        '<div class="workflow-step">'
            '<span>3</span>'
            'Explainability Review'
        '</div>'
        '<div class="workflow-arrow">&rarr;</div>'
        '<div class="workflow-step">'
            '<span>4</span>'
            'Clinician Decision'
        '</div>'
    '</div>'
)

st.html(workflow_html)


# =========================================================
# DASHBOARD TABS
# =========================================================
(
    tab_overview,
    tab_features,
    tab_ct_viewer,
    tab_3d_viewer,
    tab_prediction,
    tab_explain,
    tab_performance,
    tab_review,
    tab_report,
    tab_about,
) = st.tabs(
    [
        "Overview",
        "Nodule Characteristics",
        "CT Image Viewer",
        "3D Nodule Viewer",
        "Risk Prediction",
        "AI Explainability",
        "Model Performance",
        "Clinical Review",
        "Case Report",
        "About LungAI-CDSS",
    ]
)

# =========================================================
# TAB 1: OVERVIEW
# =========================================================
with tab_overview:
    st.subheader("Prediction Probability")

    probability_figure = go.Figure()

    probability_figure.add_trace(
        go.Bar(
            x=[
                low_probability,
                suspicious_probability,
            ],
            y=[
                "Low Risk",
                "Suspicious Risk",
            ],
            orientation="h",
            text=[
                f"{low_probability:.1%}",
                f"{suspicious_probability:.1%}",
            ],
            textposition="inside",
            marker={
                "color": [
                    "#198754",
                    "#c62828",
                ]
            },
            hovertemplate=(
                "%{y}: %{x:.1%}"
                "<extra></extra>"
            ),
        )
    )

    probability_figure.update_layout(
        height=250,
        margin={
            "l": 10,
            "r": 10,
            "t": 10,
            "b": 10,
        },
        xaxis={
            "range": [0, 1],
            "tickformat": ".0%",
            "title": "Predicted probability",
        },
        yaxis={
            "title": "",
        },
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        probability_figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    st.subheader("Model Confidence")

    gauge_colour = (
        "#c62828"
        if prediction == 1
        else "#198754"
    )

    confidence_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={
                "suffix": "%",
                "font": {
                    "size": 38,
                },
            },
            title={
                "text": f"{predicted_label} Prediction",
                "font": {
                    "size": 18,
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                },
                "bar": {
                    "color": gauge_colour,
                    "thickness": 0.30,
                },
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#d9e4ef",
                "steps": [
                    {
                        "range": [0, 60],
                        "color": "#f4f5f7",
                    },
                    {
                        "range": [60, 80],
                        "color": "#fff3cd",
                    },
                    {
                        "range": [80, 100],
                        "color": "#e8f4fc",
                    },
                ],
                "threshold": {
                    "line": {
                        "color": "#15283a",
                        "width": 4,
                    },
                    "thickness": 0.75,
                    "value": confidence * 100,
                },
            },
        )
    )

    confidence_gauge.update_layout(
        height=290,
        margin={
            "l": 30,
            "r": 30,
            "t": 50,
            "b": 20,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#15283a",
        },
    )

    st.plotly_chart(
        confidence_gauge,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    left_col, right_col = st.columns(
        [1.1, 1]
    )

    with left_col:
        st.subheader(
            "Risk Assessment"
        )

        if prediction == 1:
            st.error(
                "Suspicious-risk nodule identified"
            )
        else:
            st.success(
                "Low-risk nodule identified"
            )

        st.progress(
            float(confidence)
        )

        st.write(
            f"**Low-risk probability:** "
            f"{low_probability:.2%}"
        )

        st.write(
            f"**Suspicious-risk probability:** "
            f"{suspicious_probability:.2%}"
        )

    with right_col:
        st.subheader(
            "Dataset Reference"
        )

        st.write(
            f"**Recorded dataset group:** "
            f"{actual_group}"
        )

        if pd.notna(mean_malignancy):
            st.write(
                "**Mean radiologist malignancy score:** "
                f"{float(mean_malignancy):.2f}"
            )

        st.info(
            "The target used in this study is derived from "
            "radiologist-assessed malignancy scores and should "
            "not be interpreted as biopsy-confirmed cancer."
        )


# =========================================================
# TAB 2: NODULE CHARACTERISTICS
# =========================================================
with tab_features:
    st.subheader(
        "CT-Derived Nodule Characteristics"
    )

    display_features = pd.DataFrame(
        {
            "Feature": feature_names,
            "Value": [
                selected_row.get(
                    feature,
                    None,
                )
                for feature in feature_names
            ],
        }
    )

    st.dataframe(
        display_features,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# TAB 3: CT IMAGE VIEWER
# =========================================================
with tab_ct_viewer:
    st.subheader(
        "CT Image Viewer"
    )

    st.write(
        "Review the thoracic CT scan associated with the selected "
        "patient. The viewer automatically opens near the approximate "
        "centre of the selected radiologist-annotated nodule."
    )

    st.caption(
        "The CT image is displayed for research demonstration and "
        "clinical context. The prediction model uses structured "
        "radiologist-derived nodule features rather than raw CT pixels."
    )

    try:
        with st.spinner(
            f"Loading CT images for {selected_patient}..."
        ):
            (
                ct_volume,
                nodule_centroid,
                annotation_count,
            ) = load_ct_case(
                selected_patient,
                selected_nodule,
            )

        centre_row = int(
            round(nodule_centroid[0])
        )

        centre_column = int(
            round(nodule_centroid[1])
        )

        centre_slice = int(
            round(nodule_centroid[2])
        )

        centre_row = int(
            np.clip(
                centre_row,
                0,
                ct_volume.shape[0] - 1,
            )
        )

        centre_column = int(
            np.clip(
                centre_column,
                0,
                ct_volume.shape[1] - 1,
            )
        )

        centre_slice = int(
            np.clip(
                centre_slice,
                0,
                ct_volume.shape[2] - 1,
            )
        )

        (
            info_col1,
            info_col2,
            info_col3,
            info_col4,
        ) = st.columns(4)

        with info_col1:
            st.metric(
                "CT Slices",
                ct_volume.shape[2],
            )

        with info_col2:
            st.metric(
                "Image Size",
                f"{ct_volume.shape[0]} × "
                f"{ct_volume.shape[1]}",
            )

        with info_col3:
            st.metric(
                "Nodule Centre Slice",
                centre_slice + 1,
            )

        with info_col4:
            st.metric(
                "Radiologist Annotations",
                annotation_count,
            )

        st.markdown(
            "### Viewer Controls"
        )

        control_col1, control_col2 = st.columns(
            2
        )

        with control_col1:
            slice_index = st.slider(
                "CT slice",
                min_value=0,
                max_value=ct_volume.shape[2] - 1,
                value=centre_slice,
                step=1,
                key=(
                    f"ct_slice_slider_"
                    f"{selected_patient}_"
                    f"{selected_nodule}"
                ),
                help=(
                    "Move through the axial CT slices. "
                    "The initial position is the approximate "
                    "nodule centre."
                ),
            )

        with control_col2:
            window_choice = st.selectbox(
                "Display window",
                [
                    "Lung window",
                    "Mediastinal window",
                    "Wide window",
                ],
                key=(
                    f"ct_window_choice_"
                    f"{selected_patient}_"
                    f"{selected_nodule}"
                ),
            )

        if window_choice == "Lung window":
            window_centre = -600
            window_width = 1500

        elif window_choice == "Mediastinal window":
            window_centre = 40
            window_width = 400

        else:
            window_centre = -500
            window_width = 2000

        option_col1, option_col2 = st.columns(
            2
        )

        with option_col1:
            show_marker = st.checkbox(
                "Show approximate nodule marker",
                value=True,
                key=(
                    f"ct_show_marker_"
                    f"{selected_patient}_"
                    f"{selected_nodule}"
                ),
            )

        with option_col2:
            crop_size = st.slider(
                "Nodule zoom size",
                min_value=30,
                max_value=120,
                value=70,
                step=10,
                key=(
                    f"ct_crop_size_"
                    f"{selected_patient}_"
                    f"{selected_nodule}"
                ),
            )

        selected_slice = ct_volume[
            :,
            :,
            slice_index,
        ]

        display_slice = apply_ct_window(
            selected_slice,
            centre=window_centre,
            width=window_width,
        )

        full_col, crop_col = st.columns(
            [1.35, 1]
        )

        with full_col:
            st.markdown(
                "### Full Axial CT Slice"
            )

            full_figure, full_axis = plt.subplots(
                figsize=(7, 7)
            )

            full_axis.imshow(
                display_slice,
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            if show_marker:
                full_axis.scatter(
                    centre_column,
                    centre_row,
                    s=130,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=2,
                )

                full_axis.axhline(
                    centre_row,
                    color="red",
                    linewidth=0.8,
                    alpha=0.65,
                )

                full_axis.axvline(
                    centre_column,
                    color="red",
                    linewidth=0.8,
                    alpha=0.65,
                )

            full_axis.set_title(
                f"{selected_patient} — Slice "
                f"{slice_index + 1}/"
                f"{ct_volume.shape[2]}"
            )

            full_axis.axis(
                "off"
            )

            st.pyplot(
                full_figure,
                use_container_width=True,
            )

            plt.close(
                full_figure
            )

        with crop_col:
            st.markdown(
                "### Nodule Region"
            )

            row_start = max(
                centre_row - crop_size,
                0,
            )

            row_end = min(
                centre_row + crop_size,
                ct_volume.shape[0],
            )

            column_start = max(
                centre_column - crop_size,
                0,
            )

            column_end = min(
                centre_column + crop_size,
                ct_volume.shape[1],
            )

            cropped_slice = display_slice[
                row_start:row_end,
                column_start:column_end,
            ]

            crop_figure, crop_axis = plt.subplots(
                figsize=(6, 6)
            )

            crop_axis.imshow(
                cropped_slice,
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            if show_marker:
                crop_axis.scatter(
                    centre_column - column_start,
                    centre_row - row_start,
                    s=150,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=2,
                )

            crop_axis.set_title(
                "Magnified nodule region"
            )

            crop_axis.axis(
                "off"
            )

            st.pyplot(
                crop_figure,
                use_container_width=True,
            )

            plt.close(
                crop_figure
            )

        distance_from_centre = abs(
            slice_index - centre_slice
        )

        if distance_from_centre == 0:
            st.success(
                "The viewer is currently displaying the approximate "
                "central slice of the selected nodule."
            )

        else:
            st.info(
                f"The current slice is "
                f"{distance_from_centre} slice(s) "
                "away from the approximate nodule centre."
            )

        with st.expander(
            "CT image and annotation details"
        ):
            st.write(
                f"**Patient ID:** "
                f"{selected_patient}"
            )

            st.write(
                f"**Selected nodule:** "
                f"{selected_nodule}"
            )

            st.write(
                f"**Volume dimensions:** "
                f"{ct_volume.shape}"
            )

            st.write(
                "**Approximate centroid "
                "(row, column, slice):** "
                f"({centre_row}, "
                f"{centre_column}, "
                f"{centre_slice})"
            )

            st.write(
                "**Number of radiologist annotations:** "
                f"{annotation_count}"
            )

            st.write(
                "**Display window centre:** "
                f"{window_centre} HU"
            )

            st.write(
                "**Display window width:** "
                f"{window_width} HU"
            )

    except Exception as exc:
        st.error(
            "The CT scan could not be displayed."
        )

        st.exception(
            exc
        )

        st.info(
            "Confirm that pylidc.conf points to the full "
            "LIDC-IDRI dataset and that the selected patient "
            "has corresponding DICOM files."
        )


# =========================================================
# TAB 4: INTERACTIVE 3D NODULE VIEWER
# =========================================================
with tab_3d_viewer:
    st.subheader(
        "Interactive 3D Nodule Viewer"
    )

    st.write(
        "This viewer reconstructs the selected nodule from the "
        "consensus of the available radiologist annotations."
    )

    st.caption(
        "The displayed surface represents an annotation-derived "
        "research visualisation. It is not an automated segmentation "
        "and must not be interpreted as pathology-confirmed tumour extent."
    )

    control_col1, control_col2 = st.columns(
        2
    )

    with control_col1:
        consensus_level = st.slider(
            "Radiologist consensus level",
            min_value=0.25,
            max_value=1.00,
            value=0.50,
            step=0.25,
            key=(
                f"consensus_level_"
                f"{selected_patient}_"
                f"{selected_nodule}"
            ),
            help=(
                "At 0.50, at least half of the available "
                "radiologist annotations must include a voxel."
            ),
        )

    with control_col2:
        mesh_opacity = st.slider(
            "Surface opacity",
            min_value=0.20,
            max_value=1.00,
            value=0.85,
            step=0.05,
            key=(
                f"mesh_opacity_"
                f"{selected_patient}_"
                f"{selected_nodule}"
            ),
        )

    render_3d = st.button(
        "Generate 3D Nodule",
        type="primary",
        use_container_width=True,
        key=(
            f"render_3d_"
            f"{selected_patient}_"
            f"{selected_nodule}"
        ),
    )

    if render_3d:
        try:
            with st.spinner(
                "Generating the 3D consensus nodule surface..."
            ):
                (
                    vertices,
                    faces,
                    mask_shape,
                    annotation_count,
                    individual_mask_count,
                    approximate_volume_mm3,
                ) = load_nodule_3d_mesh(
                    selected_patient,
                    selected_nodule,
                    consensus_level,
                )

            mesh_figure = go.Figure(
                data=[
                    go.Mesh3d(
                        x=vertices[:, 1],
                        y=vertices[:, 0],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        opacity=mesh_opacity,
                        color="#d62828",
                        flatshading=False,
                        lighting={
                            "ambient": 0.45,
                            "diffuse": 0.75,
                            "specular": 0.35,
                            "roughness": 0.55,
                            "fresnel": 0.15,
                        },
                        lightposition={
                            "x": 100,
                            "y": 200,
                            "z": 150,
                        },
                        hovertemplate=(
                            "X: %{x:.2f} mm<br>"
                            "Y: %{y:.2f} mm<br>"
                            "Z: %{z:.2f} mm"
                            "<extra></extra>"
                        ),
                    )
                ]
            )

            mesh_figure.update_layout(
                height=650,
                margin={
                    "l": 0,
                    "r": 0,
                    "t": 45,
                    "b": 0,
                },
                title={
                    "text": (
                        f"{selected_patient} — "
                        f"Nodule {selected_nodule}"
                    ),
                    "x": 0.5,
                },
                scene={
                    "xaxis_title": "Column (mm)",
                    "yaxis_title": "Row (mm)",
                    "zaxis_title": "Slice depth (mm)",
                    "aspectmode": "data",
                    "camera": {
                        "eye": {
                            "x": 1.5,
                            "y": 1.5,
                            "z": 1.2,
                        }
                    },
                    "bgcolor": "#f7f9fc",
                },
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(
                mesh_figure,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                },
            )

            (
                info_col1,
                info_col2,
                info_col3,
                info_col4,
            ) = st.columns(4)

            with info_col1:
                st.metric(
                    "Radiologist Annotations",
                    annotation_count,
                )

            with info_col2:
                st.metric(
                    "Consensus Level",
                    f"{consensus_level:.0%}",
                )

            with info_col3:
                st.metric(
                    "Mesh Vertices",
                    f"{len(vertices):,}",
                )

            with info_col4:
                st.metric(
                    "Approx. Volume",
                    f"{approximate_volume_mm3:.1f} mm³",
                )

            with st.expander(
                "3D reconstruction details"
            ):
                st.write(
                    "**Consensus-mask dimensions:** "
                    f"{mask_shape}"
                )

                st.write(
                    "**Individual masks combined:** "
                    f"{individual_mask_count}"
                )

                st.write(
                    "**Surface triangles:** "
                    f"{len(faces):,}"
                )

                st.write(
                    "**Approximate consensus volume:** "
                    f"{approximate_volume_mm3:.2f} mm³"
                )

                st.write(
                    "The surface was generated using the "
                    "marching-cubes algorithm and displayed as "
                    "an interactive triangular mesh."
                )

            st.success(
                "Drag to rotate the nodule, scroll to zoom and use "
                "the Plotly toolbar to reset the camera or save an image."
            )

        except Exception as exc:
            st.error(
                "The 3D nodule reconstruction could not be generated."
            )

            st.exception(
                exc
            )

            st.info(
                "Try reducing the consensus level or confirm that the "
                "selected nodule has radiologist contour annotations."
            )

    else:
        st.info(
            "Select the desired consensus level and click "
            "'Generate 3D Nodule' to create the reconstruction."
        )


# =========================================================
# TAB 5: RISK PREDICTION
# =========================================================
with tab_prediction:
    st.subheader(
        "Explainable Prediction Summary"
    )

    st.markdown(
        "### Clinical Interpretation"
    )

    st.write(
        interpretation
    )

    st.markdown(
        "### Decision-Support Recommendation"
    )

    st.write(
        recommendation
    )

    st.caption(
        "The recommendation is generated for prototype "
        "demonstration only and does not constitute an "
        "automated diagnosis."
    )


# =========================================================
# TAB 6: AI EXPLAINABILITY
# =========================================================
with tab_explain:
    st.subheader(
        "Explainable AI Analysis"
    )

    st.write(
        "This section shows how the model arrived at its "
        "prediction. SHAP provides feature-level explanations "
        "for the selected nodule, helping the clinician understand "
        "which characteristics increased or reduced the predicted risk."
    )

    try:
        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            case_imputed
        )

        case_display = pd.DataFrame(
            case_imputed,
            columns=feature_names,
        )

        st.markdown(
            "### Patient-Specific SHAP Explanation"
        )

        fig, _ax = plt.subplots(
            figsize=(9, 4.8)
        )

        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[0],
            case_display.iloc[0],
            show=False,
        )

        st.pyplot(
            fig,
            use_container_width=False,
        )

        plt.close(
            fig
        )

        st.caption(
            "Red feature contributions increase the predicted "
            "suspicious-risk score, while blue contributions reduce it."
        )

        st.info(
            "Features pushing the prediction towards suspicious risk "
            "increase the model output, while features pushing it "
            "towards low risk reduce it."
        )

    except Exception as exc:
        st.error(
            "SHAP explanation could not be generated: "
            f"{exc}"
        )


# =========================================================
# TAB 7: MODEL PERFORMANCE
# =========================================================
with tab_performance:
    st.subheader(
        "Machine Learning Model Performance"
    )

    st.write(
        "This page compares the four supervised machine-learning "
        "models evaluated during the development of LungAI-CDSS."
    )

    if performance_df.empty:
        st.warning(
            "The model performance results file could not be loaded."
        )

    else:
        required_summary_columns = {
            "Model",
            "Accuracy",
            "ROC-AUC",
        }

        if not required_summary_columns.issubset(
            performance_df.columns
        ):
            st.error(
                "The performance file does not contain all required "
                f"columns: {sorted(required_summary_columns)}"
            )

        else:
            display_columns = [
                column
                for column in [
                    "Model",
                    "Accuracy",
                    "Precision",
                    "Recall/Sensitivity",
                    "Specificity",
                    "F1-score",
                    "ROC-AUC",
                ]
                if column in performance_df.columns
            ]

            display_df = performance_df[
                display_columns
            ].copy()

            numeric_columns = display_df.select_dtypes(
                include="number"
            ).columns

            display_df[numeric_columns] = display_df[
                numeric_columns
            ].round(4)

            best_auc_row = performance_df.loc[
                performance_df["ROC-AUC"].idxmax()
            ]

            best_accuracy_row = performance_df.loc[
                performance_df["Accuracy"].idxmax()
            ]

            (
                metric_col1,
                metric_col2,
                metric_col3,
            ) = st.columns(3)

            with metric_col1:
                st.metric(
                    "Selected Dashboard Model",
                    "XGBoost",
                )

            with metric_col2:
                st.metric(
                    "Highest ROC-AUC Model",
                    str(best_auc_row["Model"]),
                    f'{best_auc_row["ROC-AUC"]:.3f}',
                )

            with metric_col3:
                st.metric(
                    "Highest Accuracy Model",
                    str(best_accuracy_row["Model"]),
                    f'{best_accuracy_row["Accuracy"]:.3f}',
                )

            st.markdown(
                "### Model Comparison Table"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "### Comparative Performance Chart"
            )

            chart_metrics = [
                metric
                for metric in [
                    "Accuracy",
                    "Precision",
                    "Recall/Sensitivity",
                    "Specificity",
                    "F1-score",
                    "ROC-AUC",
                ]
                if metric in performance_df.columns
            ]

            if chart_metrics:
                long_performance = performance_df.melt(
                    id_vars="Model",
                    value_vars=chart_metrics,
                    var_name="Metric",
                    value_name="Score",
                )

                comparison_figure = go.Figure()

                for model_name in (
                    long_performance["Model"].unique()
                ):
                    model_data = long_performance[
                        long_performance["Model"]
                        == model_name
                    ]

                    comparison_figure.add_trace(
                        go.Bar(
                            name=model_name,
                            x=model_data["Metric"],
                            y=model_data["Score"],
                            text=model_data["Score"].map(
                                lambda value: (
                                    f"{value:.3f}"
                                )
                            ),
                            textposition="auto",
                        )
                    )

                comparison_figure.update_layout(
                    barmode="group",
                    height=470,
                    yaxis={
                        "range": [0, 1],
                        "tickformat": ".0%",
                        "title": "Performance score",
                    },
                    xaxis={
                        "title": "",
                    },
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.02,
                        "xanchor": "center",
                        "x": 0.5,
                    },
                    margin={
                        "l": 20,
                        "r": 20,
                        "t": 80,
                        "b": 20,
                    },
                )

                st.plotly_chart(
                    comparison_figure,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                    },
                )

            confusion_columns = {
                "True Negative",
                "False Positive",
                "False Negative",
                "True Positive",
            }

            xgb_rows = performance_df[
                performance_df["Model"] == "XGBoost"
            ]

            if (
                confusion_columns.issubset(
                    performance_df.columns
                )
                and not xgb_rows.empty
            ):
                st.markdown(
                    "### XGBoost Confusion Matrix"
                )

                xgb_row = xgb_rows.iloc[0]

                confusion_values = [
                    [
                        int(
                            xgb_row[
                                "True Negative"
                            ]
                        ),
                        int(
                            xgb_row[
                                "False Positive"
                            ]
                        ),
                    ],
                    [
                        int(
                            xgb_row[
                                "False Negative"
                            ]
                        ),
                        int(
                            xgb_row[
                                "True Positive"
                            ]
                        ),
                    ],
                ]

                confusion_figure = go.Figure(
                    data=go.Heatmap(
                        z=confusion_values,
                        x=[
                            "Predicted Low Risk",
                            "Predicted Suspicious Risk",
                        ],
                        y=[
                            "Actual Low Risk",
                            "Actual Suspicious Risk",
                        ],
                        text=confusion_values,
                        texttemplate="%{text}",
                        colorscale="Blues",
                        showscale=False,
                    )
                )

                confusion_figure.update_layout(
                    height=420,
                    xaxis_title="Predicted class",
                    yaxis_title="Actual class",
                    margin={
                        "l": 20,
                        "r": 20,
                        "t": 30,
                        "b": 20,
                    },
                )

                st.plotly_chart(
                    confusion_figure,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                    },
                )

            st.info(
                "Model selection should not rely on accuracy alone. "
                "Recall/sensitivity is particularly important because "
                "a false-negative result represents a suspicious nodule "
                "that the model fails to identify."
            )


# =========================================================
# TAB 8: CLINICAL REVIEW
# =========================================================
with tab_review:
    st.subheader(
        "Human-in-the-Loop Review"
    )

    clinician_decision = st.radio(
        "Clinician decision",
        [
            "Accept AI prediction",
            "Request further review",
            "Override AI prediction",
        ],
        key=(
            f"clinician_decision_"
            f"{selected_patient}_"
            f"{selected_nodule}"
        ),
    )

    clinician_notes = st.text_area(
        "Clinician notes",
        placeholder=(
            "Enter observations, justification "
            "or follow-up actions..."
        ),
        height=160,
        key=(
            f"clinician_notes_"
            f"{selected_patient}_"
            f"{selected_nodule}"
        ),
    )

    if st.button(
        "Record Review",
        use_container_width=True,
        key=(
            f"record_review_"
            f"{selected_patient}_"
            f"{selected_nodule}"
        ),
    ):
        st.success(
            "Clinician review recorded for this "
            "demonstration session."
        )

        st.write(
            f"**Decision:** "
            f"{clinician_decision}"
        )

        if clinician_notes.strip():
            st.write(
                f"**Notes:** "
                f"{clinician_notes}"
            )


# =========================================================
# TAB 9: DOWNLOADABLE CASE REPORT
# =========================================================
with tab_report:
    st.subheader(
        "Case Report Export"
    )

    malignancy_text = (
        f"{float(mean_malignancy):.2f}"
        if pd.notna(mean_malignancy)
        else "Not available"
    )

    report_text = f"""
LungAI-CDSS Clinical Decision Support Report
============================================

Generated: {datetime.now().strftime("%d %B %Y, %H:%M")}

CASE INFORMATION
----------------
Patient ID: {selected_patient}
Nodule Number: {selected_nodule}

AI RISK ASSESSMENT
------------------
AI Prediction: {predicted_label}
Model Confidence: {confidence:.1%}

Low-Risk Probability: {low_probability:.2%}
Suspicious-Risk Probability: {suspicious_probability:.2%}

DATASET REFERENCE
-----------------
Recorded Dataset Group: {actual_group}
Mean Radiologist Malignancy Score: {malignancy_text}

CLINICAL INTERPRETATION
-----------------------
{interpretation}

DECISION-SUPPORT RECOMMENDATION
-------------------------------
{recommendation}

IMPORTANT NOTICE
----------------
This report was generated by LungAI-CDSS, an academic research
prototype. It has not been clinically validated and must not be
used as an independent diagnosis or treatment recommendation.
All outputs require review by a qualified healthcare professional.
"""

    st.write(
        "Download a text summary of the selected case, "
        "prediction and decision-support recommendation."
    )

    st.download_button(
        label="Download Case Report",
        data=report_text,
        file_name=(
            f"LungAI_CDSS_"
            f"{selected_patient}_"
            f"Nodule_"
            f"{selected_nodule}.txt"
        ),
        mime="text/plain",
        use_container_width=True,
    )

# =========================================================
# TAB 10: ABOUT LUNGAI-CDSS
# =========================================================
with tab_about:
    st.subheader("About LungAI-CDSS")

    st.write(
        """
        LungAI-CDSS is an explainable artificial intelligence clinical
        decision support system prototype developed for an MSc dissertation.
        The system is designed to support the assessment of CT-derived lung
        nodule characteristics by presenting machine-learning predictions,
        model explanations, CT image visualisation, three-dimensional nodule
        reconstruction and clinician review tools within a single interface.
        """
    )

    st.info(
        "LungAI-CDSS is an academic research prototype. It has not been "
        "clinically validated and must not be used for independent diagnosis, "
        "treatment planning or direct patient care."
    )

    st.markdown("### Project Overview")

    overview_col1, overview_col2 = st.columns(2)

    with overview_col1:
        st.markdown(
            """
            **Prototype name:** LungAI-CDSS  
            **System type:** Explainable AI Clinical Decision Support System  
            **Clinical focus:** CT-based lung nodule malignancy-risk prediction  
            **Research approach:** Design Science Research  
            **Primary model:** XGBoost  
            **Programming language:** Python  
            **Application framework:** Streamlit
            """
        )

    with overview_col2:
        st.markdown(
            """
            **Dataset:** LIDC-IDRI  
            **CT scans:** 1,018  
            **Extracted nodule records:** 2,651  
            **Prediction classes:** Low Risk and Suspicious Risk  
            **Global explainability:** SHAP  
            **Local explainability:** LIME  
            **Image processing:** pylidc and pydicom
            """
        )

    st.markdown("### Dissertation Aim")

    st.write(
        """
        The aim of this dissertation is to develop and evaluate an explainable
        AI-based clinical decision support prototype for CT-based lung nodule
        malignancy-risk prediction using the LIDC-IDRI dataset. The study seeks
        to combine predictive performance with model transparency and a
        clinician-facing interface that supports human interpretation and
        oversight.
        """
    )

    st.markdown("### Research Objectives")

    st.markdown(
        """
        1. To prepare and analyse CT-derived lung nodule characteristics from
           the LIDC-IDRI dataset.

        2. To train and compare supervised machine-learning models for binary
           lung nodule risk classification.

        3. To identify the most suitable predictive model using accuracy,
           precision, recall, specificity, F1-score and ROC-AUC.

        4. To apply SHAP and LIME to provide global and local explanations of
           model predictions.

        5. To design and implement a clinician-facing dashboard that presents
           predictions, probabilities, explanations, CT images, 3D nodule
           visualisation and clinician review options.

        6. To evaluate the prototype as a transparent research artefact rather
           than an autonomous diagnostic system.
        """
    )

    st.markdown("### System Architecture")

    architecture_col1, architecture_col2, architecture_col3 = st.columns(3)

    with architecture_col1:
        st.markdown(
            """
            #### Data Layer

            - LIDC-IDRI CT scans
            - Radiologist annotations
            - DICOM image files
            - Extracted nodule features
            - Binary risk labels
            """
        )

    with architecture_col2:
        st.markdown(
            """
            #### AI and Explainability Layer

            - Logistic Regression
            - Random Forest
            - XGBoost
            - LightGBM
            - SHAP
            - LIME
            """
        )

    with architecture_col3:
        st.markdown(
            """
            #### Decision-Support Layer

            - Patient and nodule selection
            - Risk prediction
            - Probability display
            - CT image viewer
            - 3D nodule viewer
            - Clinician review
            - Case report export
            """
        )

    st.markdown("### Model Performance Summary")

    if performance_df.empty:
        st.warning(
            "Model performance information is unavailable because "
            "model_performance_results.csv could not be loaded."
        )
    else:
        xgb_rows = performance_df[
            performance_df["Model"].astype(str).str.strip() == "XGBoost"
        ]

        if xgb_rows.empty:
            st.warning(
                "The performance file does not contain an XGBoost result."
            )
        else:
            xgb_result = xgb_rows.iloc[0]

            performance_col1, performance_col2, performance_col3 = st.columns(3)

            with performance_col1:
                st.metric(
                    "XGBoost Accuracy",
                    f'{float(xgb_result["Accuracy"]):.1%}',
                )

            with performance_col2:
                st.metric(
                    "XGBoost ROC-AUC",
                    f'{float(xgb_result["ROC-AUC"]):.3f}',
                )

            with performance_col3:
                recall_column = (
                    "Recall/Sensitivity"
                    if "Recall/Sensitivity" in xgb_result.index
                    else None
                )

                if recall_column is not None:
                    st.metric(
                        "XGBoost Sensitivity",
                        f'{float(xgb_result[recall_column]):.1%}',
                    )
                else:
                    st.metric(
                        "XGBoost Sensitivity",
                        "Not available",
                    )

    st.markdown("### Key Prototype Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown(
            """
            - Live patient and nodule selection
            - XGBoost risk prediction
            - Low-risk and suspicious-risk probabilities
            - Confidence gauge
            - CT slice navigation
            - Lung, mediastinal and wide display windows
            - Approximate nodule-centre marker
            """
        )

    with feature_col2:
        st.markdown(
            """
            - Interactive 3D consensus nodule reconstruction
            - SHAP patient-specific explanations
            - Model performance comparison
            - Human-in-the-loop clinician review
            - Clinician notes
            - Downloadable case report
            - Research-use warning and governance disclaimer
            """
        )

    st.markdown("### Important Methodological Clarification")

    st.warning(
        """
        The binary target label used by LungAI-CDSS is derived from the mean
        radiologist malignancy rating in LIDC-IDRI. It therefore represents
        radiologist-assessed suspicious risk and should not be interpreted as
        biopsy-confirmed lung cancer diagnosis.
        """
    )

    st.markdown("### Limitations")

    st.markdown(
        """
        - The system was developed and tested using LIDC-IDRI and has not been
          externally validated using an independent clinical dataset.

        - The malignancy-risk labels are derived from radiologist assessments
          rather than confirmed histopathology for every nodule.

        - The prototype does not integrate with live hospital systems,
          electronic health records or Picture Archiving and Communication
          Systems.

        - The CT and 3D viewers provide contextual visualisation only. The
          XGBoost prediction is based on structured radiologist-derived nodule
          characteristics rather than raw CT pixels.

        - The clinician-review function records decisions only within the
          current demonstration session unless a persistent database is added.

        - Clinical usability, safety, fairness and governance require further
          evaluation with qualified healthcare professionals.
        """
    )

    st.markdown("### Ethical and Governance Position")

    st.write(
        """
        The prototype adopts a human-in-the-loop approach. The AI output is
        intended to support, not replace, professional judgement. Clinicians
        retain responsibility for interpreting the prediction alongside the
        patient's wider clinical information. Future deployment would require
        formal clinical validation, information-governance approval, medical
        device assessment, cybersecurity review and continuous performance
        monitoring.
        """
    )

    st.markdown("### Technology Stack")

    technology_table = pd.DataFrame(
        {
            "Component": [
                "User interface",
                "Machine-learning model",
                "Data handling",
                "Explainability",
                "CT and annotation processing",
                "3D reconstruction",
                "Interactive charts",
                "Model persistence",
            ],
            "Technology": [
                "Streamlit",
                "XGBoost",
                "pandas and NumPy",
                "SHAP and LIME",
                "pylidc and pydicom",
                "scikit-image marching cubes",
                "Plotly and Matplotlib",
                "joblib",
            ],
        }
    )

    st.dataframe(
        technology_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Prototype Information")

    st.markdown(
    """
    **Researcher:** Samuel IDOWU  
    **Programme:** MSc Health Analytics and Technologies  
    **University:** University of Greater Manchester  
    **Supervisor:** Mogul Ibtisam  
    **Prototype version:** 1.0  
    **Development year:** 2026
    """
)
# =========================================================
# FOOTER
# =========================================================
st.divider()

footer_html = (
    '<div class="dashboard-footer">'
        '<strong>LungAI-CDSS Version 1.0</strong><br>'
        'MSc Dissertation Research Prototype<br>'
        'Developed using Python, Streamlit, XGBoost, '
        'SHAP, LIME and pylidc<br>'
        '© 2026'
    '</div>'
)

st.html(footer_html)