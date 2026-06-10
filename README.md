<img width="1254" height="1254" alt="a7493fbc-b106-4ed0-9fdd-b6a968279998" src="https://github.com/user-attachments/assets/d0821ec6-dccb-46d6-b4fd-e7752d8951d1" /># Accessibility-Aware Urban Street Analysis Using Street View Imagery

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15.0-black?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Modern%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An automated, computer-vision-powered pipeline that estimates sidewalk geometry, detects pedestrian obstacles, and evaluates urban accessibility using publicly available Google Street View imagery. By combining zero-shot semantic segmentation (SAM 3), projective geometry, and multi-view feature alignment (LoFTR), the system reconstructs continuous metric maps of sidewalks to assist inclusive urban planning and wheelchair navigability assessment.

---

## Application Interface & Visualizations

### 1. Web Application: Colored Accessibility Segments

<img width="2458" height="1592" alt="accessibility-walkability_score (3)" src="https://github.com/user-attachments/assets/3c2058cb-ec98-4229-a262-2b98262bbb37" />


### 2. Sidewalk Processing Pipeline

<table>
<tr>
<td align="center" width="33%">

**SAM 3 Segmentation Mask**

<br>

<img src="https://github.com/user-attachments/assets/a0657997-eef8-4242-ac8b-367a173255d5" width="300">

<br>

*Zero-shot multi-class masks*

</td>

<td align="center" width="33%">

**Rectified Sidewalk Strip & Obstacles**

<br>

<img src="https://github.com/user-attachments/assets/ec064ab8-2af6-45d2-86b3-5d66e559d6df" width="180">

<br>

*Top-down metric projection*

</td>

<td align="center" width="33%">

**Multi-View Stitching via LoFTR**

<br>

<img src="https://github.com/user-attachments/assets/ec56365d-76ed-45f7-8f26-af850ce88104" width="180">

<br>

*Aligned consecutive tiles*

</td>
</tr>
</table>

---

## Key Contributions & Innovations

* **90° Side-Facing Street View Analysis**: Rather than relying solely on traditional road-facing images, this project introduces the use of side-facing views ($90^\circ$ left/right). Side-facing views directly capture the pedestrian corridor, significantly reducing occlusions and providing cleaner boundary views for high-accuracy sidewalk width estimation.
* **No Custom Training Required**: Leverages the zero-shot capabilities of the **Segment Anything Model 3 (SAM 3)** to segment sidewalks, roads, and a wide variety of obstacles (bollards, trees, parked vehicles, street furniture) straight out of the box.
* **Rigorous Geometric Rectification**: Implements pinhole camera back-projection and fan distortion rectification to recover metric (real-world) dimensions from 2D perspective pixels.
* **Texture-Robust Alignment**: Integrates the **Local Feature Transformer (LoFTR)** to resolve semi-dense correspondences on low-texture or repetitive concrete patterns, enabling seamless stitching of consecutive street imagery.

---

## System Workflow

<img width="1254" height="1254" alt="a7493fbc-b106-4ed0-9fdd-b6a968279998" src="https://github.com/user-attachments/assets/636141fe-cadc-43e8-bd4d-0ebcd97aab89" />


---

## Mathematical Foundations & Geometry

To convert pixel coordinates into physical meters without depth sensors, the pipeline relies on projective geometry:

### 1. Sidewalk Boundary Fitting & Horizon Estimation
For each side boundary, we extract edge pixels and fit lines using **RANSAC** to exclude transient outliers (e.g. leaves, pedestrians):
$$\text{Left Boundary: } y = a_L x + b_L$$
$$\text{Right Boundary: } y = a_R x + b_R$$

The horizon (vanishing point row coordinate $y_{vp}$) is calculated as the intersection of these two lines:
$$y_{vp} = \frac{a_L b_R - a_R b_L}{a_L - a_R}$$

### 2. Forward/Backward Pinhole Rectification
For longitudinal views (looking down the street), distance $z$ along the ground plane corresponds to the distance from the vanishing point in pixels:
$$z = \frac{f_x \cdot h}{y - y_{vp}}$$
Where:
* $h$: Camera height (typically $2.0\,\text{m}$ for Google Street View vehicles).
* $f_x$: Camera focal length in pixels, calculated from the Field of View (FOV): $f_x = \frac{W_{\text{img}}}{2 \tan(\text{FOV} / 2)}$.
* $y$: Pixel row coordinate.
* $y_{vp}$: Estimated vanishing point row (horizon).

### 3. Side-Facing Fan Rectification
Side-facing views experience "fan distortion" due to perspective projection where the foreground sidewalk is wider (occupying more pixels) than the background. The pipeline rectifies this fan-shaped quadrilateral into a uniform metric rectangle by:
1. Interpolating uniform spacing between the detected left and right boundary curves.
2. Uniformly resampling the region using a horizontal width scaling factor and an along-walk scale mapping.

---

## Accessibility Scoring Metrics

Once the rectified metric sidewalk strip is built, accessibility is quantified using two indexes:

### 1. Walkability Score
Measures overall walking quality by counting severe width reductions (narrowings or bottleneck drops):
$$\Delta W \ge 0.60\,\text{m}$$
Letting $D$ be the number of significant width-drop events along a segment:
$$\text{Walkability Score} = \frac{1}{1 + D}$$
* *Interpretation*: High score ($1.0$) means a clean, continuous width; low scores signal frequent bottlenecks.

### 2. Wheelchair Accessibility Score
Evaluates the physical viability for wheelchair navigation based on the minimum obstacle-free clear path width $w_{\text{min}}$:
$$\text{Wheelchair Score} = \begin{cases} 
1.0 & \text{if } w_{\text{min}} \ge 0.65\,\text{m} \\ 
0.0 & \text{if } w_{\text{min}} < 0.65\,\text{m} 
\end{cases}$$
*(This threshold corresponds to the minimal width required for standard manual wheelchairs to pass).*

### 3. Accessibility Classification
Segments are grouped into three distinct usability categories:
* **Accessible (Green)**: $\text{Score} \ge 0.66$
* **Marginal (Yellow)**: $0.33 \le \text{Score} < 0.66$
* **Not Accessible (Red)**: $\text{Score} < 0.33$

---

## Project Directory Layout

The codebase is structured into modular pipelines, tracking the transition from raw APIs to frontend rendering:

| Component | Path | Description |
| :--- | :--- | :--- |
| **Sampler** | [`s1-streetview-sampler-v2/`](file:///Users/berkaybgk/Desktop/cs-related/cv-urban-accessibility-senior-project/s1-streetview-sampler-v2) | OsmAPI-based geographic network sampler & downloader (4 views per point). |
| **Inference** | [`s2-inference-pipeline/`](file:///Users/berkaybgk/Desktop/cs-related/cv-urban-accessibility-senior-project/s2-inference-pipeline) | Zero-shot segmentation executor using Segment Anything (SAM 3). |
| **Visualization** | [`s3-visualization-pipeline/`](file:///Users/berkaybgk/Desktop/cs-related/cv-urban-accessibility-senior-project/s3-visualization-pipeline) | Evaluates point-by-point width estimation & obstacle anchoring. |
| **PC Width** | [`s3.5-point-cloud-width-estimation/`](file:///Users/berkaybgk/Desktop/cs-related/cv-urban-accessibility-senior-project/s3.5-point-cloud-width-estimation) | Alternative width estimation logic relying on 3D point cloud projections. |
| **Interactive App** | [`s4-interactice-web-app/`](file:///Users/berkaybgk/Desktop/cs-related/cv-urban-accessibility-senior-project/s4-interactice-web-app) | Next.js frontend showing map overlays & interactive strip creator. |
| **Strip Pipeline** | [`s5-strip-pipeline/`](file:///Users/berkaybgk/Desktop/cs-related/cv-urban-accessibility-senior-project/s5-strip-pipeline) | FastAPI service carrying out LoFTR alignment, stitching, and strip generation. |

---

## Setup & Execution

### 1. Prerequisites
Ensure you have the following installed:
* **Python**: `3.10` or `3.11` (required for model/PyTorch dependencies)
* **Node.js**: `v18+` (for the Next.js web application)
* **Google Cloud Project**: A GCS bucket for hosting image resources and segmentation masks.
* **Google Street View Static API Key**: Required for downloading image frames.

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
GCP_PROJECT_ID=your-gcp-project-id
GCS_BUCKET_NAME=your-gcs-bucket-name
GOOGLE_MAPS_API_KEY=your-google-maps-api-key
```

And configure the Next.js `.env.local` inside `s4-interactice-web-app/`:
```env
POINTS_MANIFEST_BLOB=streetview/polygon_4v/<your-run>/manifest.csv
GCS_BUCKET_NAME=your-gcs-bucket-name
GCP_PROJECT_ID=your-gcp-project-id
GCS_PREFIX_ORIGINAL=streetview
GCS_PREFIX_SEGMENTATION=v3/segmentation-results
GCS_PREFIX_VISUALIZATION=v3/visualization-results
```

---

### 3. Pipeline Execution Steps

#### Step A: Download Street View Images
Define your target bounding coordinates in `s1-streetview-sampler-v2/run.yaml` and execute the sampler:
```bash
cd s1-streetview-sampler-v2
python main.py run.yaml
```
*This downloads 4 views (forward, backward, left, right) every 10 meters.*

#### Step B: Perform Semantic Segmentation
Run the SAM 3 inference pipeline to generate masks:
```bash
cd ../s2-inference-pipeline
python main.py --config config.yaml
```

#### Step C: Start the FastAPI Strip Service
Start the backend server responsible for geometry rectification, LoFTR stitching, and obstacle mapping:
```bash
cd ../s5-strip-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn service:app --port 8000
```

#### Step D: Run the Next.js Web App
Start the interactive visualization viewer:
```bash
cd ../s4-interactice-web-app
npm install
npm run dev
```
Open `http://localhost:3000` to interact with the map, review width estimates, and open **Strip Mode** (which calls the backend service at `http://localhost:8000` to let you draw and rectify boundaries in real time).

---

## Evaluation & Results

The system was evaluated on real-world streets in Istanbul, including **Taşmektep Sokak (Caddebostan)** and sections near the **Boğaziçi University North Campus**:

* **Total Analyzed Length**: $676.3\,\text{m}$ across $16$ sidewalk segments.
* **Average Sidewalk Width**: $1.04\,\text{m}$ (Minimum) / average width variability of $0.48\,\text{m}$.
* **Average Wheelchair Passable Width**: $0.81\,\text{m}$.
* **ADA-Style Passability Rate**: $50\%$ (Half the segments failed to meet the $0.65\,\text{m}$ obstacle-free minimum clear width).
* **Width-Drop Events Detected**: $121$ significant bottlenecks.

---

## Limitations & Future Directions

* **Occlusions**: Parked cars, thick foliage, and street garbage can obscure sidewalk boundaries, leading to under-estimations or noisy boundary predictions.
* **Planar Assumptions**: The projective geometry models assume a flat ground plane and stable camera height. Steep slopes or multi-tiered sidewalks can warp metric outputs.
* **Complex Intersections**: The pipeline performs best on straight sidewalk stretches. Complex curves, crosswalk entries, and wide plazas require custom segmentation handling.
