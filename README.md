# SAM2-Integrated Annotation Platform for Biological Images

A comprehensive annotation tool for biological image stacks with integrated SAM2 (Segment Anything Model 2) functionality for semi-automatic segmentation assistance.

## Features

- **SAM2 Integration**: Point-based and box-prompt annotation with automatic propagation
- **Multi-Object Support**: Annotate multiple objects simultaneously with different colors
- **Flexible Label System**: Customizable labels with distinct colors for any annotation needs
- **Batch Processing**: Propagate annotations across multiple images automatically
- **Complete History Tracking**: All annotations are automatically saved with full history
- **ImageScope Export**: Generate XML files compatible with Aperio ImageScope
- **Flexible Image Loading**: Support for TIFF, PNG, JPG, and other common formats

## Installation

### Prerequisites

1. **Python 3.10+** is required
2. **SAM2 Model Installation**:
   - First, install SAM2 following the official instructions from [SAM2 GitHub Repository](https://github.com/facebookresearch/segment-anything-2)
   - Download the SAM2 model weights (recommended: `sam2.1_hiera_large.pt`)

### Required Folder Structure

Your SAM2 installation must have the following structure:
```
your_sam2_directory/
├── sam2/
│   ├── __init__.py
│   ├── build_sam.py
│   └── ... (other SAM2 files)
├── configs/
│   └── sam2.1/
│       └── sam2.1_hiera_l.yaml
└── sam2_model_weights/
    └── sam2.1_hiera_large.pt
```

### Installation Steps

1. **Clone this repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure SAM2 paths**:
   - Open the annotation platform script
   - Update the following paths at the top of the file:
   ```python
   SAM2_CHECKPOINT = "path/to/your/sam2_model_weights/sam2.1_hiera_large.pt"
   SAM2_CONFIG = "path/to/your/configs/sam2.1/sam2.1_hiera_l.yaml"
   ```

4. **Verify SAM2 installation**:
   ```bash
   python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2 imported successfully')"
   ```

## Customizing Labels

The platform comes with placeholder labels that should be customized for your specific use case. To modify the labels:

1. **Locate the label configuration** in the `AnnotationPipeline.__init__()` method:
   ```python
   self.label_names = ["label1", "label2", "label3", ...]  # Replace with your labels
   ```

2. **Update the colormap** to match your number of labels:
   ```python
   self.colormap = np.array([
       [R, G, B],  # Color for label1 (values 0-255)
       [R, G, B],  # Color for label2
       # ... add colors for each label
   ]) / 255.0
   ```

3. **Ensure consistency**: The number of labels must match the number of colors in the colormap

Example customization for a 5-label system:
```python
self.label_names = ["tumor", "necrosis", "stroma", "normal", "background"]
self.colormap = np.array([
    [255, 0, 0],      # Red for tumor
    [128, 0, 128],    # Purple for necrosis
    [0, 255, 0],      # Green for stroma
    [0, 0, 255],      # Blue for normal
    [128, 128, 128]   # Gray for background
]) / 255.0
```

## Usage

### Starting the Platform

```bash
python annotation_platform.py
```

### Basic Workflow

#### 1. Loading Images

1. Click **"Select Folder"** to choose your image directory
2. The platform will preview available images and show the count
3. Choose to load all images or specify a range using the start/end indices
4. Click **"Load Images"** to load the selected images

#### 2. Manual Annotation

1. **Select a Label**: Choose from your configured labels using the checkboxes
2. **Draw Annotations**: 
   - Draw polygons directly on the image for manual annotations
   - Polygons are automatically colored based on the selected label
   - Edge width is set to 10 for finalized annotations

#### 3. SAM2-Assisted Annotation

##### Understanding SAM2's Working ROI

SAM2 operates within a **limited working region** (the green rectangle) rather than the entire image. This design choice prevents distortion and excessive downsampling when large image areas are passed to SAM2's neural network, which expects inputs of a specific size. By constraining SAM2 to a smaller region, the platform maintains segmentation quality and accuracy.

##### SAM2 Workflow

1. **Draw a Rectangle**: Draw a rectangular region of interest around the area you want to segment
2. **Activate SAM2**: Click **"SAM2 Seg"** to convert the rectangle into a SAM2 working area (turns green)
3. **Add Prompts** (within the green rectangle only):
   - **Box Prompts** (recommended): Draw rectangles around objects - these are typically much more effective than point prompts
   - **Positive Points** (green): Click inside objects you want to segment
   - **Negative Points** (red): Click outside objects to exclude areas
   - **Combined Prompts**: Use both box and point prompts together for optimal results

##### Multi-Object Segmentation

The platform supports sophisticated multi-object segmentation within a single working ROI:

1. **Object ID System**: Each object is assigned a unique ID (1, 2, 3, etc.)
2. **Color Coding**: Objects are automatically assigned different colors (cyan, magenta, yellow, orange, purple, pink, brown, gray)
3. **Workflow for Multiple Objects**:
   - Start with Object ID 1 (default)
   - Add prompts (box/points) for the first object
   - Change **"SAM2 Object ID"** to 2
   - Add prompts for the second object
   - Continue for additional objects as needed
   - All objects are processed simultaneously when you click Propagate

4. **Object Management**:
   - Each object maintains its own set of prompts
   - Objects can have different combinations of box and point prompts
   - The platform shows active objects and their prompt counts
   - Objects are segmented independently but processed in the same batch

##### Propagation

**Propagation** means extending the segmentation from your annotated frame(s) to adjacent frames in the image stack:

- **Forward Propagation**: Applies segmentation to subsequent images
- **Backward Propagation**: Applies segmentation to previous images
- **Number of Images**: Adjust using **"Propagate N images"** spinbox (1-100)
- SAM2 uses temporal consistency to track objects across frames

#### 4. Segmentation Refinement

Refinement allows you to improve SAM2's segmentation results by adding additional prompts:

1. **Initial Propagation**: First run propagation to see initial results
2. **Enter Refinement Mode**: Click **"Refine Seg"**
3. **Add Refinement Prompts**:
   - Navigate to frames where segmentation needs improvement
   - Add additional **point prompts only** (positive or negative)
   - Box prompts cannot be added during refinement
4. **Re-propagate**: Click the same propagation button you used initially (Forward or Backward)
5. **Iterative Process**: You can refine multiple times until satisfied

**Important**: Refinement prompts are combined with the original prompts, so SAM2 considers all annotations together when re-computing the segmentation.

#### 5. Finalizing Annotations

- **Finalize SAM2 Seg**: Converts SAM2 results to permanent annotations with the selected label and removes the working area
- **Undo SAM2 Seg**: Removes all annotations from the current SAM2 session
- **Convert Unused Rectangles**: Converts untracked rectangles to manual box annotations (useful for cases where you want box annotations without triggering SAM2)

### Saving Your Work

**Important**: Save your work regularly! The platform can handle large datasets, but regular saving prevents data loss:

- Click **"Save Annotations"** frequently
- New saves overwrite previous saves in the same directory
- Saving is fast and has no downside
- Recommended: Save after completing each major annotation task

The platform saves:
- Individual JSON files for each annotated image
- Complete annotation history
- Summary statistics
- ImageScope-compatible XML files

### ImageScope XML Export

The platform automatically generates XML files compatible with Aperio ImageScope:
- One XML file per image with annotations
- All configured label categories included (even if empty)
- Proper color mapping for each label
- Files are saved in `imagescope_xmls/` subdirectory

## Platform Capabilities

### Annotation Types

- **Manual ROIs**: Hand-drawn polygons with label-based coloring
- **SAM2 Results**: AI-assisted segmentations that can be refined
- **Working ROIs**: Temporary regions for SAM2 processing (always green)
- **Box Prompts**: Rectangle guides for SAM2 segmentation

### Advanced Features

- **Session Tracking**: Each SAM2 segmentation session is tracked separately
- **Shape Modification**: Edit existing annotations by selecting and modifying
- **Batch Operations**: Process multiple images simultaneously
- **Annotation History**: Complete record of all annotation actions
- **Debug Tools**: Built-in debugging for SAM2 state and session tracking

## Tips for Effective Use

### SAM2 Best Practices

1. **Box Prompts First**: Start with box prompts as they're typically more effective than points
2. **Combine Prompt Types**: Use box prompts for initial selection, then refine with points
3. **Working ROI Size**: Keep the working ROI reasonably sized to maintain segmentation quality
4. **Multi-Object Strategy**: Process all objects in one propagation for efficiency

### Refinement Strategy

1. **Review Results**: After initial propagation, review all frames
2. **Identify Problem Areas**: Note frames where segmentation is inaccurate
3. **Strategic Point Placement**: Add points at boundaries or missed regions
4. **Sparse Refinement**: Often, refining a few key frames improves the entire sequence

### General Tips

1. **Save Frequently**: Save after each major annotation milestone
2. **Label Consistency**: Always verify the correct label is selected before annotating
3. **Rectangle Management**: Rectangles remain available for SAM2 until explicitly converted or used
4. **Performance**: For large image stacks, consider processing in batches

## Troubleshooting

### Common Issues

1. **SAM2 Import Error**: Ensure SAM2 is properly installed and paths are correctly configured
2. **CUDA/GPU Issues**: The platform automatically falls back to CPU if GPU is unavailable
3. **Mixed Dimension Shapes**: The platform automatically converts all shapes to 3D coordinates
4. **Large Images**: Consider using the range selection feature to load images in batches

### Debug Tools

- **"Show Summary"**: Displays annotation statistics
- **"Debug SAM2 State"**: Shows current SAM2 configuration and active objects
- **"Debug Sessions"**: Displays session tracking information

## System Requirements

- **RAM**: 8GB minimum, 16GB+ recommended for large image stacks
- **GPU**: CUDA-capable GPU recommended for faster SAM2 processing (CPU fallback available)
- **Storage**: Sufficient space for images and annotation files
- **Display**: 1920x1080 or higher resolution recommended
