# Eggsactly
#### Egg Counting Tool for *Drosophila melanogaster* Egg-Laying Assays

## Introduction

This software project provides a web-based application to facilitate the automated counting of eggs laid by *Drosophila melanogaster* during egg-laying assays. In these experiments, specialized chambers are used to house individual female flies while allowing researchers to study egg deposition patterns under different conditions.

**Chamber Configurations:**
- **Standard Two-Choice Chambers:**  
  These feature a grid of plastic enclosures where each enclosure contains a female fly. Each enclosure is flanked by two agarose areas—one on each of two opposite sides—with distinct chemical compositions. Multiple grid configurations are supported (e.g., 6x4 grid, 3x5 grid, etc.).
- **Large Four-Choice Chambers:**  
  Some assays use a larger chamber with a square footprint that contains four small circular agarose areas arranged in a diamond formation.

**How the Tool Works:**
- **Image Segmentation:**  
  The user uploads an overhead photograph of the assay plate. A neural network object detector identifies key reference points (such as the circular holes in the plate) to scale and segment the image into individual egg-laying chambers.
- **Egg Detection and Counting:**  
  For each segmented agarose area, the image is cropped and processed through a second object detector tuned to recognize eggs. The detected eggs are outlined, and counts are generated.
- **Flexibility:**  
  In addition to the supported chamber types, users can define custom masks to accommodate other chamber configurations, as long as the egg sizes in the images are similar to those used to train the egg-detection model.
- **User Interaction:**  
  The results are presented via a zoomable and pannable web interface where users can inspect the segmentation and detection outputs, manually correct counts if necessary, and download the final results as an annotated image or CSV file.

This README details the installation, configuration, and usage of the tool, as well as instructions for reproducing a demo assay.

Note: the revision of this code as used in the article ["Natural non-coding _pumilio_ variants retune value-coding interneurons to bias _Drosophila_ oviposition choices"](https://www.biorxiv.org/content/10.1101/2025.03.07.641729v2) is available as a [Figshare archive](https://figshare.com/articles/software/Eggsactly/30581984).

## System Requirements

### Software Dependencies

- **Python:**  
  - Version **3.9** (this is the only version that has been tested)

- **Key Python Packages:**  
  The core functionality of the app depends on several key packages, including:
  - **Flask** (v2.2.2) – for the web server
  - **PyTorch** (v1.12.0) – for the object-detection models (requires CUDA)
  - **Additional Essentials:**  
    - csbdeep==0.7.2
    - Flask-Dance==6.2.0
    - Flask-Login==0.6.2
    - Flask-Session==0.4.0
    - Flask-SocketIO==5.3.1
    - Flask-SQLAlchemy==3.0.2
    - numpy==1.23.3
    - opencv-python==4.10.0
    - Pillow==9.2.0

  For a complete list of the top-level dependencies (with version numbers), please refer to the provided [`requirements.txt`](./requirements.txt).

- **Environment Reproducibility:**  
  To replicate the full environment—including non-Python dependencies and the correct CUDA toolkit version—we provide **two** Conda environment files:
  
  - **For NVIDIA GPUs with CUDA 11.6:**  
    Use [`environment.cuda11.6.yml`](./environment.cuda11.6.yml)
  
  - **For NVIDIA GPUs with CUDA 11.0:**  
    Use [`environment.cuda11.0.yml`](./environment.cuda11.0.yml)

### Operating Systems

- **Linux:**  
  - The application has been tested on **Ubuntu 20.04** and **22.04**.
- **Windows:**  
  - Supported in theory, though not formally tested.
- **macOS:**  
  - **Not supported** for GPU-dependent functionality because the app relies on CUDA, and macOS does not support CUDA. *(Note: A CPU-only mode is not available.)*

### Hardware Requirements

- **GPU:**  
  - **Mandatory:** A CUDA-enabled **NVIDIA GPU** is required to run the egg-detection model.  
  - **Tested GPUs:** NVIDIA GeForce RTX 3060 and NVIDIA GeForce GTX TITAN.
  - **Important:** The application does **not** support CPU-only operation. If your system does not have a CUDA-capable GPU, the app will not function.
  
- **CUDA Toolkit:**  
  - The application has been verified with **CUDA 11.6** (with an RTX 3060) and **CUDA 11.0** (with a GTX TITAN). Ensure that you use the appropriate environment file for your GPU.
  
- **Disk Space:**  
  - Approximately **500 MB** is recommended for installation, including the model files and demo data.
  
- **RAM:**  
  - **8 GB** or more is recommended for optimal performance.


## Installation Guide
Follow these steps to install and run the egg-counting tool on your local machine.

### Step 1: Clone the repository
Clone the respository and navigate to its directory:
```bash
git clone https://github.com/rcalfredson/Eggsactly.git
cd Eggsactly
```
### Step 2: Set Up the Environment
You can choose one of two methods:
**Option A: Using Conda (Recommended for GPU/CUDA support)**
Use one of the provided environment files based on your GPU's CUDA version:
  - **For NVIDIA GPUs with CUDA 11.6:**
    ```bash
    conda env create -f environment.cuda11.6.yml
    ```
  - **For NVIDIA GPUs with CUDA 11.0:**
    ```bash
    conda env create -f environment.cuda11.0.yml
    ```
Activate the environment:
```bash
conda activate eggcounting
```

**Option B: Using a Python Virtual Environment**
Alternatively, set up a Python virtual environment:
```bash
python -m venv my_env
source my_env/bin/activate   # On Windows: my_env\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a `.env` file in the root directory of the project. This file must contain settings for both the server and the GPU worker(s). Below is an example configuration:
```ini
# Common settings (for both server and GPU workers)
EGG_COUNTING_BACKEND_TYPE=sql         # How to store data; 'sql' is currently supported.
SQL_ADDR_TYPE=sqlite                  # Options: 'sqlite' (recommended), 'shortname', or 'ip_addr'
SECRET_KEY=your_secret_key_here       # Required for Flask session management (encrypts cookies)
NUM_GPU_WORKERS=1                     # Number of GPU workers supporting the server
GPU_WORKER_TIMEOUT=30                 # Max seconds the server waits for a GPU worker response

# Optional settings for Google Cloud MySQL or OAuth:
# GOOGLE_SQL_CONN_NAME=your_conn_name_here
# GOOGLE_SQL_DB_DVT_IP=your_db_ip_here
# GOOGLE_SQL_DB_PASSWORD=your_db_password_here
# GOOGLE_CLIENT_ID=your_google_client_id_here
# GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# GPU worker–specific settings
MAIN_SERVER_URI=http://localhost:5000  # URI of the egg-counting server
PRIVATE_KEY_PATH=project/auth/gpu_worker_1_id_rsa.pem  # Path to the GPU worker's key (see Step 5)
GPU_WORKER_RECONNECT_ATTEMPT_DELAY=10  # Delay (in seconds) between reconnection attempts
```

### Step 4: Set Up the Database
If you are using SQLite (recommended for quick setup), run the following commands from a Python shell:
```python
from project import create_app, db
app = create_app()
with app.app_context():
    db.create_all()
```

For Google Cloud MySQL, please refer to the additional setup instructions in the documentation.

### Step 5: Generate GPU Worker Public/Private Key Pairs
Generate a new key pair to secure communication between the server and the GPU worker. These keys are used to sign and verify requests: the GPU worker uses its private key to generate an access token (JWT) that accompanies each POST request to the server, and the server uses the corresponding public key (located in project/auth) to authenticate these requests.
1. Run the key pair generator:
   ```bash
   python project/scripts/rsa_keypair_gen.py
   ```
   The script will output a private key (starting with `-----BEGIN PRIVATE KEY-----` and ending with `-----END PRIVATE KEY-----`).
2. Copy this output into a file named according to your `PRIVATE_KEY_PATH` (for example, `project/auth/gpu_worker_1_id_rsa.pem`).
3. Set the appropriate file permissions:
   ```bash
   chmod +x project/auth/gpu_worker_1_id_rsa.pem
   ```
4. Generate the corresponding public key:
   ```bash
   openssl rsa -in project/auth/gpu_worker_1_id_rsa.pem -pubout -out project/auth/gpu_worker_1_id_rsa.pub
   ```
   Ensure that the public key is stored in the `project/auth` directory using the naming convention `gpu_worker_{int}_id_rsa.pub` (where `{int}` ranges from 1 to `NUM_GPU_WORKERS`).

### Step 6: Start the Server and GPU Worker
Open two terminal windows (or tabs) and run the following:
**Terminal 1 (Server):**
```bash
python -m project.server --host 0.0.0.0 --port 5000
```
**Terminal 2 (GPU Worker):**
```bash
python -m project.gpu_backend.worker
```

### Estimated Installation Time
The full installation process should take approximately 15–30 minutes on a standard desktop computer.

## Data and Model Acquisition

To facilitate testing and demonstration of the egg-counting tool, we provide two key resources hosted on Figshare:

### Egg Detection Model

A pre-trained SplineDist UNet model optimized for detecting *Drosophila melanogaster* eggs is available:

- **Title:** SplineDist UNet Weights for Drosophila melanogaster Egg Detection  
- **Figshare Link:** [SplineDist UNet Weights for Drosophila melanogaster Egg Detection](https://figshare.com/articles/software/SplineDist_UNet_Weights_for_i_Drosophila_melanogaster_i_Egg_Detection/28382495?file=52244816)  
- **Description:**  
  This file contains the pre-trained weights for the UNet backbone of a SplineDist-based object detection model (a variant of StarDist). The model is optimized for detecting Drosophila melanogaster eggs in backlit images of transparent gel substrates and was trained for 400 epochs.
- **Instructions:**  
  Download the `.pth` file and place it in the `project/models/` directory. The model expects the configuration file located at `project/configs/unet_backbone_rand_zoom.json`.

### Demo Dataset

A small, self-contained demo dataset is provided to help you test and visualize the tool’s capabilities:

- **Title:** Egg Counting Tool Demo Dataset  
- **Figshare Link:** [Egg Counting Tool Demo Dataset](https://figshare.com/articles/dataset/Egg_Counting_Tool_Demo_Dataset/28382828)  
- **Description:**  
  The dataset is organized into two directories:
  - **Inputs:**  
    The `inputs/` directory contains three raw backlit images (`image_1.JPG`, `image_2.JPG`, and `image_3.JPG`) that serve as sample inputs for the tool.
  - **Outputs:**  
    The `outputs/` directory includes the corresponding results for each input image:
    - **Annotated Images (`*_annotated.JPG`):**  
      These images display the detected egg outlines and segmented agarose areas, along with egg counts per region.
    - **Egg Count Spreadsheets (`*_counts.csv`):**  
      These CSV files list the numerical egg counts for each agarose region.
- **Usage:**  
  Upload an image from the `inputs/` directory to the web app and process it. Then, compare the resulting annotated image and CSV file with the ones in the `outputs/` directory to verify the tool’s performance.

## Demo

This section provides a step-by-step walkthrough to demonstrate the functionality of the egg-counting tool using the provided demo dataset. It covers the entire process—from uploading images, reviewing segmentation, performing egg counting, to downloading and verifying results.

For additional details or advanced usage instructions (such as manual segmentation adjustments), please refer to the appropriate section in the documentation.

### Step 1: Access the Web Application

Open your web browser and navigate to the server URL (for example, `http://localhost:5000`). You should see a simple interface with a button labeled **"Choose images to upload"**.

![A snapshot of the home page showing the upload button.](doc/imgs/image_upload_button.png)

### Step 2: Upload Demo Images

Click the **"Choose images to upload"** button. A system file selection dialog will appear.  
- Use shift-click (or your system’s equivalent) to select all three demo images from the demo dataset's `inputs/` folder.  
- Click **"Open"** to upload the selected images to the application.

![An image of the file selection dialog with multiple demo images selected.](doc/imgs/demo_image_upload_selection.png)

### Step 3: Inspect Segmentation Results

After the images are uploaded, the tool automatically processes them to segment each image into its individual agarose areas.  
- The segmented results are displayed in an interactive viewer where you can zoom, pan, and step through the processed images.  
- If any segmentation error occurs (for example, if an image isn’t correctly segmented), the interface allows you to specify the chamber type and manually place key reference points. (This manual adjustment is intended for advanced usage and may be covered separately.)

![A close-up view of an image in the interactive viewer showing the segmented agarose areas and outlined eggs.](doc/imgs/agarose_segmentation_example.png)

### Step 4: Perform Egg Counting

Once you are satisfied with the segmentation:
- **Count Eggs (Single Image):** Click the **"Count eggs"** button to process only the image currently displayed in the viewer.
- **Count Eggs (All Images):** Click the **"Count eggs (all images)"** button to process the entire batch of uploaded images.

While counting is in progress, you may see progress indicators that display the status of the operation.

![View of egg counts and outlines on an analyzed image.](doc/imgs/segmented_eggs_example.png)

### Step 5: Download the Results

After egg counting completes, you have two options for exporting the results:
- **Download Egg Counts:** Click this button to download a CSV file that lists the egg counts arranged according to the layout of the agarose areas.
- **Download Annotated Images:** Click this button to download a ZIP archive containing full-resolution versions of the processed images with segmented outlines and egg counts superimposed.

**Note:**  
If you process images as a single batch, the resulting CSV file will aggregate counts for all images in one file. Although the demo dataset provides separate CSV files for each image (to keep outputs modular), you can still compare the numerical results to verify the tool's performance.

![Sample previews of the CSV file and one of the annotated images.](doc/imgs/egg_counts_downloads_example.png)

### Expected Runtime
On a standard desktop computer, processing an image (i.e., the automatic segmentation and egg detection) typically takes about 7-8 seconds. This does not include any additional time required for user inspection or manual adjustments.


## Usage Instructions

While the **Demo** section provides an overview of the basic workflow, this section details how to use the tool on your own data, including advanced features for handling cases where automatic segmentation does not yield satisfactory results.

### Basic Workflow

For most images, the processing workflow is identical to the demo:
1. **Upload your image(s):** Use the **"Choose images to upload"** button to select and upload one or more images.
2. **Review segmentation:** The tool will automatically segment each image into its constituent agarose areas and display the results in the interactive viewer.
3. **Perform egg counting:** Click **"Count eggs"** (for a single image) or **"Count eggs (all images)"** to run the egg detection.
4. **Download outputs:** Retrieve the results as a CSV file and/or a ZIP archive containing annotated images.

*For a complete walkthrough of these basic steps, please refer to the Demo section above.*

### Advanced Segmentation Adjustments

In cases where the automatic segmentation does not work as expected, the tool will indicate an error, as shown below, and provide options for manual intervention.

![View of the "Image analysis error" that occurs when automatic segmentation does not work as expected](doc/imgs/image_analysis_error.png)

#### 1. Manual Chamber Type Specification

If an image’s segmentation fails and the chamber type is one of the natively supported types (e.g., "6 rows x 4 cols", "5 rows x 3 cols", "optogenetic", or "large"):
- **Select the appropriate chamber type** from the dropdown menu.
- You will then be presented with an interface where you can manually adjust the segmentation.  
  - **Manual Alignment:**  
    Drag three points on the image to form two connected line segments. These segments should intersect at a designated reference point (typically in the upper left-hand corner) and align along the specific vertical and horizontal reference lines for the selected chamber type.
    
    ![A view of the manual alignment interface showing the three draggable points and reference lines overlaying the image.](doc/imgs/manual_chamber_alignment.png)

#### 2. Custom Masks

Custom masks are available for images that do not conform to any of the natively supported chamber types, or if you prefer to define your own regions. Note that custom masks are available as an alternative option even if automatic segmentation *succeeded* for your image.
- **Select "Custom masks"** from the chamber type dropdown.
- Two mask definition modes are available:
  
  **a. Grid Mode:**  
  - Define a series of horizontal and vertical lines that form a grid over the image.
  - Select any of the resulting rectangular areas as regions for egg counting.
  
    ![An image with grid lines overlaid and one or more rectangular regions highlighted.](doc/imgs/custom_mask_gridlines.png)
  
  **b. Freehand Mode:**  
  - Draw rectangular areas freely on the image to indicate where egg counts should be performed.
  

    ![An image with freehand-drawn rectangles outlining regions of interest.](doc/imgs/custom_mask_freehand.png)
  
- **Saving Custom Masks:**  
  - If you are not logged in, the last custom mask is saved locally via cookies.
  - If you are logged in, you can save multiple named sets of custom masks and load them on future sessions.

### Reviewing and Correcting Results

After segmentation and egg detection:
- **Inspect the Results:** Use the interactive viewer to review the segmented regions and egg outlines.
- **Manual Corrections:**  
  If you notice any inaccuracies (in either segmentation or egg counting), use the provided correction tools to adjust the counts or refine the region boundaries before downloading the final outputs.

### Troubleshooting and Tips

- **Consistency:**  
  If you frequently work with images that require manual adjustments, consider saving your custom mask settings to ensure consistent processing across similar images.

- **Choosing the Appropriate Mask Mode:**  
  - Use **Grid Mode** when your regions are arranged in a regular grid.
  - Use **Freehand Mode** when the layout of regions is irregular or does not conform to a grid pattern.

- **Getting Help:**  
  If you encounter issues or have questions, please create an issue on our [GitHub repository](https://github.com/rcalfredson/Eggsactly/issues) for support.


