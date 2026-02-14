"""
Unified Annotation Pipeline with SAM2 Integration
================================================

A comprehensive tool for annotating image stacks with integrated SAM2 
functionality for automated segmentation assistance.

Features include:
- SAM2 integration with point-based annotation
- Automatic rectangle-to-square ROI conversion
- Multi-object annotation support
- Batch propagation across multiple images
- Complete annotation history tracking
- Save/load functionality

Requirements:
- Python 3.10+
- napari
- magicgui
- numpy
- scikit-image
- tifffile (for image I/O)
- pandas (for annotation tracking)
- opencv-python (for image processing)
- torch (for SAM2)
- SAM2 model and weights
"""

import napari
from napari.layers import Image, Labels, Shapes, Points
from magicgui import magicgui, magic_factory
from magicgui.widgets import Container, PushButton, SpinBox, CheckBox, Label as LabelWidget, FileEdit, ComboBox
import numpy as np
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set, Union, cast
import tifffile
from skimage.draw import polygon
from dataclasses import dataclass, asdict
import cv2
import torch
import time
import warnings
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import sys
import os
import shutil
from PIL import Image as PILImage

warnings.filterwarnings('ignore')

# Force OpenCV to use non-Qt backend to avoid conflicts with napari
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'

try:
    from qtpy.QtCore import Signal
except ImportError:
    try:
        from qtpy.QtCore import pyqtSignal as Signal
    except ImportError:
        from PySide2.QtCore import Signal

from typing import Dict, List, Tuple, Optional, Set, Union, cast, Any


# ============================================================================
# SAM2 MODEL CONFIGURATION
# ============================================================================
# TODO: Update these paths to match your SAM2 installation
SAM2_CHECKPOINT = r"C:\Users\Florin\OneDrive - Johns Hopkins\Documents\segmentation_mask_correction_pipeline\sam2_model_weights\sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


@dataclass
class AnnotationRecord:
    """Record of a single annotation"""
    timestamp: str
    image_index: int
    annotation_type: str  # 'roi', 'point', 'sam2_result'
    coordinates: List[Tuple[float, float]]
    object_id: Optional[int] = None
    metadata: Optional[Dict] = None


@dataclass
class PointAnnotation:
    """Record of a point annotation for SAM2"""
    coordinates: Tuple[float, float]  # (y, x) in global coordinates
    point_type: str  # 'positive' or 'negative'
    roi_local_coords: Tuple[float, float]  # (y, x) in ROI-local coordinates
    image_index: int


class AnnotationPipeline:
    """Main pipeline class for image annotation with SAM2 integration"""
    
    def __init__(self):

        # Define colormap and labels (reuse from original)
        self.colormap = np.array([
            [214, 212, 161],  # 1 bone
            [247, 184, 67],   # 2 brain + spinal cord
            [136, 232, 95],   # 3 eye
            [140, 13, 13],    # 4 heart
            [38, 27, 166],    # 5 lungs
            [13, 125, 11],    # 6 GI track 
            [179, 50, 108],   # 7 liver
            [228, 235, 131],  # 8 spleen
            [156, 96, 235],   # 9 pancreas
            [46, 190, 230],   # 10 kidney
            [150, 255, 245],  # 11 mesokidney
            [254, 222, 255],  # 12 collagen
            [235, 154, 108],  # 13 ear
            [255, 255, 255],  # 14 nontissue
            [9, 64, 116],     # 15 thymus
            [255, 255, 74],   # 16 thyroid
            [178, 178, 0],    # 17 bladder
            [214, 212, 161],  # 18 skull
            [54, 83, 89]      # 19 spleen2
        ]) / 255.0  # Normalize to 0-1 range
        
        self.label_names = ["bone", "brain", "eye", "heart", "lungs", "GI", "liver", 
                        "spleen", "pancreas", "kidney", "mesokidney", "collagen", 
                        "ear", "nontissue", "thymus", "thyroid", "bladder", "skull", "spleen2"]
        

        # FLEXIBLE: Automatically determine number of labels
        self.num_labels = len(self.label_names)
        
        # Validate colormap matches labels
        if len(self.colormap) != self.num_labels:
            raise ValueError(f"Colormap has {len(self.colormap)} colors but there are {self.num_labels} labels")
    
        # Add label selection state
        self.selected_label: int = 1  # Default to label 1 (bone)

        # Initialize state variables
        self.image_stack: Optional[np.ndarray] = None
        self.current_index: int = 0
        
        # Annotation tracking
        self.annotation_history: List[AnnotationRecord] = []
        self.all_annotations: Dict[int, List[Dict]] = {}  # {image_idx: [annotation_dicts]}
        
        # File paths
        self.image_folder: Optional[Path] = None
        self.save_directory: Optional[Path] = None
        self.original_filenames: List[str] = []
        
        # Napari viewer and layers
        self.viewer: Optional[napari.Viewer] = None
        self.image_layer: Optional[Image] = None
        self.shapes_layer: Optional[Shapes] = None
        
        # SAM2 integration - Point layers
        self.positive_points_layer: Optional[Points] = None
        self.negative_points_layer: Optional[Points] = None
        
        # SAM2 state
        self.current_square_roi: Optional[np.ndarray] = None
        self.current_roi_params: Optional[Dict[str, int]] = None
        self.sam2_mode: str = "idle"  # "idle", "annotation", "propagated", "refining"
        self.propagated_images: Set[int] = set()
        self.sam2_results_cache: Dict[int, List[np.ndarray]] = {}
        self.sam2_working_roi_id: Optional[int] = None
        self.point_annotations_by_object: Dict[int, List[PointAnnotation]] = {}
        self.sam2_box_prompts_by_object: Dict[int, Optional[Tuple[float, float, float, float]]] = {}
        self.active_object_ids: Set[int] = set()
        self.current_sam2_object_id: int = 1
        
        # SAM2 session tracking for selective undo
        self.sam2_session_id: int = 0  # Increments with each new SAM2 seg
        self.shape_to_session_mapping: Dict[int, int] = {}  # shape_idx -> session_id
        
        # Track annotations state
        self._last_point_snapshot: Dict[str, int] = {'positive': 0, 'negative': 0}
        self._last_shapes_snapshot: int = 0
        
        # Comprehensive annotation tracking
        self.shape_to_object_mapping: Dict[int, int] = {}
        self.shape_to_type_mapping: Dict[int, str] = {}
        self.object_to_box_shape_index: Dict[int, int] = {}
        
        # Point tracking with napari layer indices
        self.positive_point_to_object: Dict[int, int] = {}
        self.negative_point_to_object: Dict[int, int] = {}
        
        # Track which points have been processed
        self.processed_positive_points: Set[int] = set()
        self.processed_negative_points: Set[int] = set()
        
        # Add unique shape ID tracking
        self._next_shape_id = 1
        self.shape_id_to_napari_index: Dict[int, int] = {}  # shape_id -> current napari index
        self.napari_index_to_shape_id: Dict[int, int] = {}  # current napari index -> shape_id
        self.shape_id_to_type: Dict[int, str] = {}  # shape_id -> type
        self.shape_id_to_object: Dict[int, int] = {}  # shape_id -> object_id
        self.shape_id_to_session: Dict[int, int] = {}  # shape_id -> session_id
        self.shape_id_to_coordinates: Dict[int, np.ndarray] = {}  # shape_id -> original coordinates

        # Add shape modification tracking
        self.shape_selection_snapshot = {
            'selected_indices': set(),
            'shapes_data': {},  # {index: (shape_id, coordinates, metadata)}
            'snapshot_time': None
        }
        self._modifying_shapes = False  # Flag to prevent recursion

        # Initialize session tracking
        if not hasattr(self, 'sam2_session_id'):
            self.sam2_session_id = 0
        if not hasattr(self, 'shape_to_session_mapping'):
            self.shape_to_session_mapping = {}

    def preview_folders(self, image_folder: str) -> Tuple[int, List[str]]:
        """Preview folder to get image count and filenames without loading"""
        try:
            img_folder = Path(image_folder)
            
            # Supported image formats
            image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']
            
            # Get all image files
            img_files = []
            for ext in image_extensions:
                img_files.extend(img_folder.glob(ext))
                img_files.extend(img_folder.glob(ext.upper()))
            
            img_files = sorted(list(set(img_files)))
            filenames = [f.name for f in img_files]
            
            return len(img_files), filenames
        except Exception as e:
            print(f"Error previewing folder: {e}")
            return 0, []

    def load_image_folder(self, image_folder: str, start_idx: Optional[int] = None, 
                         end_idx: Optional[int] = None) -> None:
        """Load images from folder"""
        self.image_folder = Path(image_folder)
        
        # Supported image formats
        image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']
        
        # Get all image files
        img_files = []
        for ext in image_extensions:
            img_files.extend(self.image_folder.glob(ext))
            img_files.extend(self.image_folder.glob(ext.upper()))
        
        # Sort files
        img_files = sorted(list(set(img_files)))
        
        if len(img_files) == 0:
            raise ValueError("No images found in the specified folder")
        
        # Apply range selection
        total_images = len(img_files)
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = total_images
        
        # Validate range
        if start_idx < 0:
            start_idx = 0
        if end_idx > total_images:
            end_idx = total_images
        if start_idx >= end_idx:
            raise ValueError(f"Invalid range: start_idx ({start_idx}) must be less than end_idx ({end_idx})")
        
        # Select files in range
        img_files = img_files[start_idx:end_idx]
        
        # Store original filenames
        self.original_filenames = [f.name for f in img_files]
        
        print(f"Loading images {start_idx} to {end_idx-1} ({len(img_files)} images) out of {total_images} total...")
        
        # Load first image to get dimensions
        first_img = self.load_single_image(img_files[0])
        
        # Initialize image stack
        if len(first_img.shape) == 3:  # RGB
            self.image_stack = np.zeros((len(img_files), *first_img.shape), dtype=np.uint8)
        else:  # Grayscale
            self.image_stack = np.zeros((len(img_files), *first_img.shape, 3), dtype=np.uint8)
        
        # Load all images
        for i, img_file in enumerate(img_files):
            print(f"Loading {i+1}/{len(img_files)} (image {start_idx + i} in folder): {img_file.name}")
            
            img = self.load_single_image(img_file)
            
            # Handle grayscale images
            if len(img.shape) == 2:
                self.image_stack[i] = np.stack([img, img, img], axis=-1)
            elif len(img.shape) == 3 and img.shape[-1] > 3:
                self.image_stack[i] = img[..., :3]
            elif len(img.shape) == 3 and img.shape[-1] < 3:
                padding = 3 - img.shape[-1]
                self.image_stack[i] = np.pad(img, ((0, 0), (0, 0), (0, padding)), mode='constant')
            else:
                self.image_stack[i] = img
                
            # Ensure proper dtype
            if self.image_stack[i].dtype != np.uint8:
                if self.image_stack[i].dtype in [np.float32, np.float64]:
                    if self.image_stack[i].max() <= 1.0:
                        self.image_stack[i] = (self.image_stack[i] * 255).astype(np.uint8)
                    else:
                        self.image_stack[i] = np.clip(self.image_stack[i], 0, 255).astype(np.uint8)
                else:
                    self.image_stack[i] = np.clip(self.image_stack[i], 0, 255).astype(np.uint8)
        
        print(f"Loaded {len(img_files)} images: shape={self.image_stack.shape}")
    
        if self.viewer is not None:
            self.synchronize_all_mappings()


    def load_single_image(self, filepath: Path) -> np.ndarray:
        """Load a single image file"""
        try:
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError("Please install pillow: pip install pillow")
        
        # Try tifffile first for TIFF files
        if filepath.suffix.lower() in ['.tif', '.tiff']:
            try:
                img = tifffile.imread(filepath)
                return img
            except Exception:
                pass
        
        # Use PIL for all formats
        try:
            img = PILImage.open(filepath)
            
            # Handle different modes
            if img.mode == 'L':
                img_array = np.array(img)
            elif img.mode == 'RGBA':
                img_array = np.array(img)[..., :3]
            elif img.mode in ['RGB', 'BGR']:
                img_array = np.array(img)
            else:
                img = img.convert('RGB')
                img_array = np.array(img)
            
            return img_array
            
        except Exception as e:
            raise ValueError(f"Could not load image {filepath}: {e}")
    
    # ============================================================================
    # ANNOTATION MANAGEMENT
    # ============================================================================
    

    def save_annotation_with_shape_id(self, image_index: int, annotation_type: str, 
                                    coordinates: np.ndarray, shape_id: int,
                                    object_id: Optional[int] = None, 
                                    metadata: Optional[Dict] = None,
                                    label: Optional[int] = None) -> None:
        """Save annotation with shape ID for better tracking"""
        if metadata is None:
            metadata = {}
        
        # Add shape ID to metadata
        metadata['shape_id'] = shape_id
        
        # Call original save_annotation
        self.save_annotation(image_index, annotation_type, coordinates, 
                            object_id, metadata, label)


    def save_annotation(self, image_index: int, annotation_type: str, 
                    coordinates: Union[np.ndarray, List[Tuple[float, float]]], 
                    object_id: Optional[int] = None, metadata: Optional[Dict] = None,
                    label: Optional[int] = None) -> None:
        """Save an annotation to history"""
        # Convert coordinates to list format and handle 2D/3D
        if isinstance(coordinates, np.ndarray):
            if coordinates.ndim == 2 and coordinates.shape[1] == 3:
                # 3D coordinates (Z, Y, X) - extract Y, X only
                coord_list = coordinates[:, 1:].tolist()
            elif coordinates.ndim == 2 and coordinates.shape[1] == 2:
                # 2D coordinates (Y, X) - use as is
                coord_list = coordinates.tolist()
            else:
                print(f"Warning: Unexpected coordinate shape: {coordinates.shape}")
                coord_list = coordinates.tolist()
        else:
            coord_list = list(coordinates)
        
        # Use selected label if not specified
        if label is None:
            label = self.selected_label
        
        # Add label to metadata
        if metadata is None:
            metadata = {}
        metadata['label'] = label
        # FLEXIBLE: Use num_labels for validation
        metadata['label_name'] = self.label_names[label - 1] if 1 <= label <= self.num_labels else 'unknown'
        
        # Add session tracking for SAM2-related annotations
        if annotation_type in ['working_roi', 'box_prompt', 'sam2_result', 'positive_point', 'negative_point']:
            metadata['session_id'] = getattr(self, 'sam2_session_id', 0)
        
        # Create record
        record = AnnotationRecord(
            timestamp=datetime.now().isoformat(),
            image_index=image_index,
            annotation_type=annotation_type,
            coordinates=coord_list,
            object_id=object_id,
            metadata=metadata
        )
        
        self.annotation_history.append(record)
        
        # Also store in per-image structure
        if image_index not in self.all_annotations:
            self.all_annotations[image_index] = []
        
        self.all_annotations[image_index].append({
            'type': annotation_type,
            'coordinates': coord_list,
            'object_id': object_id,
            'label': label,
            'label_name': metadata['label_name'],
            'metadata': metadata
        })
        
        print(f"💾 Saved {annotation_type} annotation: img={image_index}, label={label}, coords_len={len(coord_list)}")
    

    def rgb_to_bgr_int(self, r: float, g: float, b: float) -> int:
        """Convert RGB values (0-1 range) to BGR integer for ImageScope LineColor"""
        # Convert to 0-255 range and ensure integers
        r_int = int(r * 255)
        g_int = int(g * 255)
        b_int = int(b * 255)
        
        # Pack as BGR integer: Blue + (Green << 8) + (Red << 16)
        bgr_int = b_int + (g_int << 8) + (r_int << 16)
        return bgr_int


    def create_imagescope_xml_for_image(self, image_annotations: List[Dict], 
                                    output_path: str, microns_per_pixel: float = 0.504) -> None:
        """Create ImageScope-compatible XML file for a single image"""
        import xml.etree.ElementTree as ET
        
        # Create root element
        root = ET.Element('Annotations')
        root.set('MicronsPerPixel', str(microns_per_pixel))
        
        annotation_id = 1
        region_id = 1
        
        # FLEXIBLE: Group annotations by label for all available labels
        annotations_by_label = {i+1: [] for i in range(self.num_labels)}
        
        for ann in image_annotations:
            # Only include user ROIs and finalized SAM2 results
            if ann.get('type') not in ['user_roi', 'sam2_result']:
                continue
                
            label = ann.get('label', 1)
            if 1 <= label <= self.num_labels:
                annotations_by_label[label].append(ann)
        
        # FLEXIBLE: Create annotation layer for EACH label, based on num_labels
        for label in range(1, self.num_labels + 1):
            label_name = self.label_names[label - 1]
            label_annotations = annotations_by_label[label]
            
            # Get color for this label
            label_color_rgb = self.colormap[label - 1]
            line_color = self.rgb_to_bgr_int(
                label_color_rgb[0], 
                label_color_rgb[1], 
                label_color_rgb[2]
            )
            
            # Create annotation element (even if no regions)
            annotation_elem = ET.SubElement(root, 'Annotation')
            annotation_elem.set('Id', str(annotation_id))
            annotation_elem.set('Name', label_name)
            annotation_elem.set('ReadOnly', '0')
            annotation_elem.set('NameReadOnly', '0')
            annotation_elem.set('LineColorReadOnly', '0')
            annotation_elem.set('Incremental', '0')
            annotation_elem.set('Type', '4')
            annotation_elem.set('LineColor', str(line_color))
            annotation_elem.set('Visible', '1')
            annotation_elem.set('Selected', '0')
            annotation_elem.set('MarkupImagePath', '')
            annotation_elem.set('MacroName', '')
            
            # Create regions container (even if empty)
            regions_elem = ET.SubElement(annotation_elem, 'Regions')
            
            # Add regions if this label has annotations
            if label_annotations:
                for ann in label_annotations:
                    coordinates = ann.get('coordinates', [])
                    
                    if not coordinates or len(coordinates) < 3:
                        continue
                        
                    # Create region element
                    region_elem = ET.SubElement(regions_elem, 'Region')
                    region_elem.set('Id', str(region_id))
                    region_elem.set('Type', '0')  # Polygon
                    region_elem.set('NegativeROA', '0')
                    region_elem.set('InputRegionId', '0')
                    region_elem.set('Analyze', '1')
                    region_elem.set('DisplayId', str(region_id))
                    region_elem.set('Selected', '0')
                    # REMOVED Text attribute - no label text will appear next to annotations
                    region_elem.set('Zoom', '0.0')
                    region_elem.set('ImageLocation', '')
                    region_elem.set('ImageFocus', '0')
                    
                    # Create vertices container
                    vertices_elem = ET.SubElement(region_elem, 'Vertices')
                    
                    # Add vertices
                    for vertex in coordinates:
                        if len(vertex) >= 2:
                            vertex_elem = ET.SubElement(vertices_elem, 'Vertex')
                            # ImageScope uses X, Y order
                            vertex_elem.set('X', str(float(vertex[1])))
                            vertex_elem.set('Y', str(float(vertex[0])))
                    
                    # Close polygon by adding first vertex again if needed
                    if len(coordinates) > 2:
                        first = coordinates[0]
                        last = coordinates[-1]
                        if abs(first[0] - last[0]) > 1e-6 or abs(first[1] - last[1]) > 1e-6:
                            vertex_elem = ET.SubElement(vertices_elem, 'Vertex')
                            vertex_elem.set('X', str(float(first[1])))
                            vertex_elem.set('Y', str(float(first[0])))
                    
                    region_id += 1
            
            annotation_id += 1
        
        # Write XML with explicit encoding
        tree = ET.ElementTree(root)
        
        # Pretty print
        ET.indent(tree, space="  ", level=0)
        
        # Write with explicit XML declaration
        with open(output_path, 'wb') as f:
            tree.write(f, encoding='utf-8', xml_declaration=True)
        
        return region_id - 1  # Return total number of regions

    def save_imagescope_xmls(self, save_directory: str) -> None:
        """Save ImageScope-compatible XML files for each annotated image"""
        if not self.all_annotations:
            print("No annotations to save in ImageScope format")
            return
        
        if not self.original_filenames:
            print("Warning: No original filenames available")
            return
        
        save_dir = Path(save_directory)
        imagescope_dir = save_dir / "imagescope_xmls"
        imagescope_dir.mkdir(exist_ok=True)
        
        xml_count = 0
        total_regions = 0
        
        print("\n📋 Creating ImageScope XML files...")
        
        # Process each image that has annotations
        for img_idx, annotations in sorted(self.all_annotations.items()):
            # Filter valid annotations
            valid_annotations = [
                ann for ann in annotations 
                if ann.get('type') in ['user_roi', 'sam2_result'] and
                ann.get('coordinates') and 
                len(ann.get('coordinates', [])) >= 3
            ]
            
            if not valid_annotations:
                continue
            
            # Get original filename for this image
            if img_idx < len(self.original_filenames):
                original_filename = self.original_filenames[img_idx]
                # Remove extension and add .xml
                base_name = Path(original_filename).stem
                xml_filename = f"{base_name}.xml"
            else:
                # Fallback if index is out of range
                xml_filename = f"image_{img_idx:04d}.xml"
            
            xml_path = imagescope_dir / xml_filename
            
            try:
                # Create XML for this specific image
                num_regions = self.create_imagescope_xml_for_image(
                    valid_annotations, 
                    str(xml_path)
                )
                
                xml_count += 1
                total_regions += num_regions
                
                print(f"  ✅ {xml_filename}: {num_regions} regions across 19 annotation layers")
                
            except Exception as e:
                print(f"  ❌ Error creating XML for image {img_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create summary file
        summary_path = imagescope_dir / "imagescope_export_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("ImageScope XML Export Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Export Date: {datetime.now().isoformat()}\n")
            f.write(f"XML Files Created: {xml_count}\n")
            f.write(f"Total Regions: {total_regions}\n")
            # FLEXIBLE: Report actual number of labels
            f.write(f"Annotation Layers per File: {self.num_labels} (all labels included)\n\n")
            
            f.write("Files created:\n")
            for xml_file in sorted(imagescope_dir.glob("*.xml")):
                f.write(f"  - {xml_file.name}\n")
            
            f.write("\nInstructions:\n")
            f.write("1. Place these XML files in the same directory as your slide images\n")
            f.write("2. The XML filename must match the slide filename (e.g., slide.svs and slide.xml)\n")
            f.write("3. ImageScope will automatically load the annotations when you open the slide\n")
            f.write(f"4. All {self.num_labels} annotation layers are included in each file (even if empty)\n")
            f.write("5. No text labels will appear next to the annotations\n")
            
            f.write("\nColor Mapping (Label -> BGR Integer):\n")
            # FLEXIBLE: Iterate through actual number of labels
            for i in range(self.num_labels):
                label_name = self.label_names[i]
                rgb = self.colormap[i]
                bgr_int = self.rgb_to_bgr_int(rgb[0], rgb[1], rgb[2])
                f.write(f"  {i+1}. {label_name}: {bgr_int}\n")
        
        print(f"\n✅ ImageScope XML export complete:")
        print(f"   - Created {xml_count} XML files in: {imagescope_dir}")
        print(f"   - Total regions: {total_regions}")
        print(f"   - Each file contains all 19 annotation layers")
        print(f"   - Summary saved: {summary_path}")


    def save_project(self, directory: str) -> None:
        """Save all annotations and project state INCLUDING ImageScope XMLs"""
        if self.image_stack is None:
            print("No data to save")
            return
            
        save_dir = Path(directory)
        save_dir.mkdir(exist_ok=True)
        
        # Save annotations
        annotations_dir = save_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)
        
        # Save per-image annotations
        for img_idx, annotations in self.all_annotations.items():
            if annotations:
                filename = f"image_{img_idx:04d}_annotations.json"
                with open(annotations_dir / filename, 'w') as f:
                    json.dump(annotations, f, indent=2)
        
        print(f"Saved annotations for {len(self.all_annotations)} images")
        
        # Save annotation history
        try:
            history_data = [asdict(record) for record in self.annotation_history]
            with open(save_dir / "annotation_history.json", 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Create summary
            if history_data:
                summary_df = pd.DataFrame(history_data)
                summary_df.to_csv(save_dir / "annotation_summary.csv", index=False)
                
                # Summary statistics
                summary_stats = {
                    "total_annotations": len(history_data),
                    "annotated_images": len(self.all_annotations),
                    "annotation_types": {}
                }
                
                for record in history_data:
                    ann_type = record['annotation_type']
                    summary_stats["annotation_types"][ann_type] = summary_stats["annotation_types"].get(ann_type, 0) + 1
                
                with open(save_dir / "summary_statistics.json", 'w') as f:
                    json.dump(summary_stats, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save annotation history: {e}")
        
        # Save project metadata
        try:
            metadata = {
                "image_folder": str(self.image_folder) if self.image_folder else "",
                "original_filenames": self.original_filenames,
                "total_annotations": len(self.annotation_history),
                "num_images": len(self.image_stack) if self.image_stack is not None else 0,
                "save_timestamp": datetime.now().isoformat()
            }
            with open(save_dir / "project_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
        
        # NEW: Save ImageScope-compatible XMLs (one per image)
        try:
            self.save_imagescope_xmls(str(save_dir))
        except Exception as e:
            print(f"Warning: Could not save ImageScope XMLs: {e}")
            import traceback
            traceback.print_exc()
            
        print(f"\n✅ Project saved to {save_dir}")
        print(f"- Annotations: {len(self.annotation_history)} total")
        print(f"- Annotated images: {len(self.all_annotations)}")
        print(f"- ImageScope XMLs: Created for each annotated image")
        print("\n🎉 Your annotations are safely saved!")

    # ============================================================================
    # ROI UTILITIES (kept from original)
    # ============================================================================
    
    def validate_square_roi(self, roi_vertices: np.ndarray) -> bool:
        """Validate if ROI is a square/rectangle"""
        if len(roi_vertices) != 4:
            return False
        
        v = roi_vertices
        
        # Calculate vectors for each side
        side1 = v[1] - v[0]
        side2 = v[2] - v[1]
        side3 = v[3] - v[2]
        side4 = v[0] - v[3]
        
        # Check if opposite sides are equal
        tolerance = 1e-6
        
        if not (np.abs(np.linalg.norm(side1) - np.linalg.norm(side3)) < tolerance and
                np.abs(np.linalg.norm(side2) - np.linalg.norm(side4)) < tolerance):
            return False
        
        # Check if adjacent sides are perpendicular
        if not (np.abs(np.dot(side1, side2)) < tolerance and
                np.abs(np.dot(side2, side3)) < tolerance):
            return False
        
        return True
    
    def get_rectangle_roi_params(self, roi_vertices: np.ndarray) -> Dict[str, int]:
        """Extract parameters from a rectangular ROI"""
        # Handle both 2D and 3D coordinates
        if roi_vertices.shape[1] == 3:
            y_coords = roi_vertices[:, 1]
            x_coords = roi_vertices[:, 2]
        elif roi_vertices.shape[1] == 2:
            y_coords = roi_vertices[:, 0]
            x_coords = roi_vertices[:, 1]
        else:
            raise ValueError(f"Invalid ROI shape: {roi_vertices.shape }")
        
        # Find bounding box
        min_y = int(np.min(y_coords))
        max_y = int(np.max(y_coords))
        min_x = int(np.min(x_coords))
        max_x = int(np.max(x_coords))
        
        height = max_y - min_y
        width = max_x - min_x
        
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid ROI dimensions: width={width}, height={height}")
        
        return {
            'top_left_y': min_y,
            'top_left_x': min_x,
            'height': height,
            'width': width
        }
    
    def convert_rectangle_to_square(self, rect_params: Dict[str, int]) -> Dict[str, int]:
        """Convert rectangular ROI parameters to square ROI parameters"""
        square_size = max(rect_params['height'], rect_params['width'])
        
        center_y = rect_params['top_left_y'] + rect_params['height'] // 2
        center_x = rect_params['top_left_x'] + rect_params['width'] // 2
        
        new_top_left_y = center_y - square_size // 2
        new_top_left_x = center_x - square_size // 2
        
        return {
            'top_left_y': new_top_left_y,
            'top_left_x': new_top_left_x,
            'height': square_size,
            'width': square_size
        }
    
    def validate_roi_bounds(self, roi_params: Dict[str, int]) -> bool:
        """Validate that ROI fits within image boundaries"""
        if self.image_stack is None:
            return False
            
        img_height, img_width = self.image_stack.shape[1:3]
        
        if (roi_params['top_left_y'] < 0 or 
            roi_params['top_left_x'] < 0 or
            roi_params['top_left_y'] + roi_params['height'] > img_height or
            roi_params['top_left_x'] + roi_params['width'] > img_width):
            return False
            
        return True
    
    def create_square_roi_vertices(self, square_params: Dict[str, int]) -> np.ndarray:
        """Create square ROI vertices from parameters"""
        top_left_y = square_params['top_left_y']
        top_left_x = square_params['top_left_x']
        size = square_params['height']
        
        # Create 2D vertices
        square_roi_2d = np.array([
            [top_left_y, top_left_x],
            [top_left_y, top_left_x + size],
            [top_left_y + size, top_left_x + size],
            [top_left_y + size, top_left_x]
        ], dtype=np.float64)
        
        # Convert to 3D with current image index
        z_coord = self.current_index
        z_coords = np.full((4, 1), z_coord, dtype=np.float64)
        square_roi_3d = np.hstack([z_coords, square_roi_2d])
    
        return square_roi_3d
    
    def get_square_roi_params(self, roi_vertices: np.ndarray) -> Dict[str, int]:
        """Extract parameters from a square/rectangular ROI"""
        return self.get_rectangle_roi_params(roi_vertices)
    
    def convert_to_square_roi(self, roi_params: Dict[str, int]) -> Optional[np.ndarray]:
        """Convert rectangular ROI parameters to a square ROI"""
        square_params = self.convert_rectangle_to_square(roi_params)
        
        if not self.validate_roi_bounds(square_params):
            return None
        
        return self.create_square_roi_vertices(square_params)
    
    def safely_clear_rois(self, preserve_prompts: bool = False) -> None:
        """Safely clear all ROIs and associated tracking"""
        try:
            if self.shapes_layer and len(self.shapes_layer.data) > 0:
                # Save annotations before clearing
                for idx, shape in enumerate(self.shapes_layer.data):
                    shape_type = self.shape_to_type_mapping.get(idx, 'user_roi')
                    obj_id = self.shape_to_object_mapping.get(idx)
                    
                    # Save non-working ROIs as annotations
                    if shape_type != 'working_roi':
                        self.save_annotation(
                            self.current_index,
                            shape_type,
                            shape,
                            obj_id,
                            {'saved_before_clear': True}
                        )
                
                self.shapes_layer.selected_data = set(range(len(self.shapes_layer.data)))
                self.shapes_layer.remove_selected()
                print("ROIs cleared safely")
            
            # Clear tracking
            self.shape_to_object_mapping.clear()
            self.shape_to_type_mapping.clear()
            self.object_to_box_shape_index.clear()
            self.sam2_working_roi_id = None
            
            if not preserve_prompts:
                self.sam2_box_prompts_by_object.clear()
            
            self._recompute_active_objects()
            
        except Exception as e:
            # synchronize mappings
            self.synchronize_all_mappings()
            print(f"Warning: Could not clear ROIs safely: {e}")
    
    
    def safely_add_roi(self, roi_vertices: np.ndarray, target_images: Optional[List[int]] = None, 
                is_working_roi: bool = False, object_id: Optional[int] = None, 
                label: Optional[int] = None, **kwargs) -> None:
        """Safely add ROI(s) with proper tracking and label"""

        try:
            if self.shapes_layer is None:
                print("Warning: No shapes layer available")
                return
                
            # synchronize mappings
            self.synchronize_all_mappings()

            if roi_vertices is None or len(roi_vertices) < 3:
                print(f"Warning: Invalid ROI vertices")
                return
            
            # CRITICAL FIX: Set flag to prevent auto-processing of SAM2 results
            if object_id is not None and object_id > 0:
                self._adding_sam2_results = True
                print(f"🤖 Adding SAM2 result - will prevent auto-processing as user ROI")
            
            # CRITICAL FIX: Strict validation of ROI vertices
            if not isinstance(roi_vertices, np.ndarray):
                roi_vertices = np.array(roi_vertices)
            
            # Validate and clean the array
            roi_vertices = roi_vertices.astype(np.float64)
            
            # Check for invalid values
            if not np.all(np.isfinite(roi_vertices)):
                print("Warning: ROI contains invalid coordinates (NaN/Inf)")
                return
            
            # Check for reasonable size
            if len(roi_vertices) > 1000:
                print(f"Warning: ROI has too many vertices ({len(roi_vertices)}), limiting to 1000")
                # Sample the vertices to reduce complexity
                step = len(roi_vertices) // 1000
                roi_vertices = roi_vertices[::max(1, step)]
            
            # Handle both 2D and 3D input, but always output 3D
            if roi_vertices.ndim == 2:
                if roi_vertices.shape[1] == 2:
                    # 2D coordinates - need to add Z dimension
                    if target_images is not None and len(target_images) > 0:
                        z_coord = target_images[0]
                    else:
                        z_coord = self.current_index
                    
                    # Convert to 3D
                    z_coords = np.full((len(roi_vertices), 1), z_coord, dtype=np.float64)
                    roi_vertices = np.hstack([z_coords, roi_vertices])
                elif roi_vertices.shape[1] == 3:
                    # Already 3D
                    pass
                else:
                    print(f"Warning: ROI vertices have wrong shape {roi_vertices.shape}")
                    return
            else:
                print(f"Warning: ROI vertices have wrong dimensionality {roi_vertices.ndim}")
                return
                
            # Remove any duplicate consecutive points
            if len(roi_vertices) > 1:
                diff = np.diff(roi_vertices, axis=0)
                keep_indices = np.where(np.any(np.abs(diff) > 1e-6, axis=1))[0]
                if len(keep_indices) > 0:
                    keep_indices = np.concatenate(([0], keep_indices + 1))
                    roi_vertices = roi_vertices[keep_indices]
            
            # Final validation
            if len(roi_vertices) < 3:
                print("Warning: ROI has too few unique points after cleaning")
                return
            
            # Determine color based on context
            if is_working_roi:
                edge_color = 'lime'
                edge_width = 4
            elif object_id is not None and object_id > 0:
                # SAM2 result - use object colors
                object_colors = ['cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray']
                color_idx = (object_id - 1) % len(object_colors)
                edge_color = object_colors[color_idx]
                edge_width = 2
            else:
                # User ROI - use label color
                if label is None:
                    label = self.selected_label
                label_color_rgb = self.colormap[label - 1]
                edge_color = '#{:02x}{:02x}{:02x}'.format(
                    int(label_color_rgb[0] * 255),
                    int(label_color_rgb[1] * 255), 
                    int(label_color_rgb[2] * 255)
                )
                edge_width = 3
            
            default_kwargs = {
                'shape_type': 'polygon',
                'edge_color': edge_color, 
                'edge_width': edge_width,
                'face_color': [0, 0, 0, 0]
            }
            default_kwargs.update(kwargs)
            
            # Temporarily disconnect event handler
            try:
                self.shapes_layer.events.data.disconnect(self._on_shapes_changed)
            except:
                pass
            
            starting_roi_count = len(self.shapes_layer.data)
            added_indices = []
            
            if target_images is not None and len(target_images) > 0:
                for img_idx in target_images:
                    try:
                        # Create 3D coordinates for this specific image
                        roi_for_image = roi_vertices.copy()
                        roi_for_image[:, 0] = img_idx  # Set Z coordinate to image index
                        
                        # Validate the 3D array one more time
                        if (roi_for_image.shape[1] != 3 or 
                            not np.all(np.isfinite(roi_for_image)) or
                            len(roi_for_image) < 3):
                            print(f"Warning: Invalid 3D ROI for image {img_idx}")
                            continue
                        
                        self.shapes_layer.add(roi_for_image, **default_kwargs)
                        added_indices.append(starting_roi_count + len(added_indices))
                        
                    except Exception as e:
                        # synchronize mappings
                        self.synchronize_all_mappings()
                        print(f"Warning: Could not add ROI to image {img_idx}: {e}")
                        continue
            else:
                try:
                    # Add single ROI (already 3D)
                    if (roi_vertices.shape[1] != 3 or 
                        not np.all(np.isfinite(roi_vertices)) or
                        len(roi_vertices) < 3):
                        print("Warning: Invalid 3D ROI")
                        return
                        
                    self.shapes_layer.add(roi_vertices, **default_kwargs)
                    added_indices.append(starting_roi_count)
                    
                except Exception as e:
                    # synchronize mappings
                    self.synchronize_all_mappings()
                    print(f"Warning: Could not add 3D ROI: {e}")
            
            # Reconnect event handler BEFORE updating tracking
            self.shapes_layer.events.data.connect(self._on_shapes_changed)
            
            # Update tracking and save annotations
            for idx in added_indices:
                if is_working_roi:
                    self.sam2_working_roi_id = idx
                    # Create shape ID for working ROI
                    shape_id = self._next_shape_id
                    self._next_shape_id += 1
                    
                    # Set up tracking for working ROI
                    self.shape_id_to_napari_index[shape_id] = idx
                    self.napari_index_to_shape_id[idx] = shape_id
                    self.shape_id_to_coordinates[shape_id] = roi_vertices.copy()
                    self.shape_id_to_object[shape_id] = -1
                    self.shape_id_to_type[shape_id] = 'working_roi'
                    self.shape_id_to_session[shape_id] = getattr(self, 'sam2_session_id', 0)
                    
                    print(f"🔒 Tracked SAM2 working ROI at index {idx} (session {self.sam2_session_id})")
                    
                elif object_id is not None:
                    # Create shape ID for SAM2 result
                    shape_id = self._next_shape_id
                    self._next_shape_id += 1
                    
                    # Set up tracking for SAM2 result
                    self.shape_id_to_napari_index[shape_id] = idx
                    self.napari_index_to_shape_id[idx] = shape_id
                    self.shape_id_to_coordinates[shape_id] = roi_vertices.copy()
                    self.shape_id_to_object[shape_id] = object_id
                    self.shape_id_to_type[shape_id] = 'sam2_result'
                    self.shape_id_to_session[shape_id] = getattr(self, 'sam2_session_id', 0)
                    
                    print(f"✅ Added SAM2 result at index {idx} for Object {object_id} (session {self.sam2_session_id})")
                    
                    # Save SAM2 result as annotation
                    if target_images:
                        for img_idx in target_images:
                            self.save_annotation(
                                img_idx,
                                'sam2_result',
                                roi_vertices,  # save_annotation will handle 3D->2D conversion
                                object_id,
                                {'edge_color': default_kwargs.get('edge_color'), 'shape_id': shape_id},
                                label=label or self.selected_label
                            )
                # Note: User ROIs are handled by _on_shapes_changed when not flagged as SAM2 results
            
            # Reset the flag
            if hasattr(self, '_adding_sam2_results'):
                self._adding_sam2_results = False
                
        except Exception as e:
            # synchronize mappings
            self.synchronize_all_mappings()

            print(f"Warning: Could not add ROI safely: {e}")
            import traceback
            traceback.print_exc()
            
            # Reset the flag on error
            if hasattr(self, '_adding_sam2_results'):
                self._adding_sam2_results = False
            
            try:
                if self.shapes_layer is not None:
                    self.shapes_layer.events.data.connect(self._on_shapes_changed)
            except:
                pass

    # ============================================================================
    # SAM2 INTEGRATION - kept essential functions
    # ============================================================================
    
    def _is_roi_inside_working_roi(self, roi: np.ndarray) -> bool:
        """Check if an ROI is inside the working square ROI"""
        if self.current_roi_params is None:
            return False
        
        if roi.shape[1] == 3:
            y_coords = roi[:, 1]
            x_coords = roi[:, 2]
        elif roi.shape[1] == 2:
            y_coords = roi[:, 0]
            x_coords = roi[:, 1]
        else:
            return False
        
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        
        working_top = self.current_roi_params['top_left_y']
        working_left = self.current_roi_params['top_left_x']
        working_bottom = working_top + self.current_roi_params['height']
        working_right = working_left + self.current_roi_params['width']
        
        return (working_top <= min_y < max_y <= working_bottom and
                working_left <= min_x < max_x <= working_right)
    
    def global_to_local_coords(self, global_coords: Tuple[float, float], 
                              roi_params: Dict[str, int]) -> Tuple[float, float]:
        """Convert global image coordinates to ROI-local coordinates"""
        global_y, global_x = global_coords
        local_y = global_y - roi_params['top_left_y']
        local_x = global_x - roi_params['top_left_x']
        return (local_y, local_x)
    
    def local_to_global_coords(self, local_coords: Tuple[float, float], 
                              roi_params: Dict[str, int]) -> Tuple[float, float]:
        """Convert ROI-local coordinates to global image coordinates"""
        local_y, local_x = local_coords
        global_y = local_y + roi_params['top_left_y']
        global_x = local_x + roi_params['top_left_x']
        return (global_y, global_x)
    
    def initialize_point_layers(self) -> None:
        """Initialize positive and negative point layers"""
        if self.viewer is None:
            return
            
        ndim = 3
        
        if self.positive_points_layer is None:
            self.positive_points_layer = self.viewer.add_points(
                data=np.empty((0, ndim)),
                name="Positive Points",
                size=8,
                face_color='lime',
                border_color='darkgreen',
                border_width_is_relative=False,
                border_width=1,
                symbol='o',
                ndim=ndim
            )
            self._last_positive_count = 0
            self.positive_points_layer.events.data.connect(
                lambda event: self._on_positive_points_changed()
            )
        
        if self.negative_points_layer is None:
            self.negative_points_layer = self.viewer.add_points(
                data=np.empty((0, ndim)),
                name="Negative Points", 
                size=8,
                face_color='red',
                border_color='darkred',
                border_width_is_relative=False,
                border_width=1,
                symbol='x',
                ndim=ndim
            )
            self._last_negative_count = 0
            self.negative_points_layer.events.data.connect(
                lambda event: self._on_negative_points_changed()
            )
        
        # Ensure visibility
        if self.positive_points_layer is not None:
            self.positive_points_layer.visible = True
            try:
                if self.positive_points_layer in self.viewer.layers:
                    layer_index = list(self.viewer.layers).index(self.positive_points_layer)
                    self.viewer.layers.move(layer_index, -1)
            except:
                pass
                
        if self.negative_points_layer is not None:
            self.negative_points_layer.visible = True
            try:
                if self.negative_points_layer in self.viewer.layers:
                    layer_index = list(self.viewer.layers).index(self.negative_points_layer)
                    self.viewer.layers.move(layer_index, -1)
            except:
                pass
        
        print("✅ Point layers initialized")
    
    def initialize_shapes_layer(self) -> None:
        """Initialize shapes layer with proper event handling"""
        if self.viewer is None:
            return
            
        if self.shapes_layer is not None:
            try:
                self.shapes_layer.events.data.disconnect()
                self.shapes_layer.events.selected_data.disconnect()
            except:
                pass
        
        if self.shapes_layer is None:
            # Create shapes layer with 3D coordinates from the start
            self.shapes_layer = self.viewer.add_shapes(
                data=np.empty((0, 0, 3)),  # Ensure 3D from initialization
                name="Annotations",
                edge_color="yellow",
                edge_width=3,
                face_color="transparent",
                ndim=3  # Explicitly set to 3D
            )
        
        # Connect events
        self.shapes_layer.events.data.connect(self._on_shapes_changed)
        self.shapes_layer.events.selected_data.connect(self._on_selection_changed)  # NEW
        print("✅ Shapes layer initialized with 3D coordinates")
    

    def emergency_shape_recovery(self) -> None:
        """Emergency recovery from mixed dimension shapes"""
        if self.shapes_layer is None:
            return
        
        print("🚨 Running emergency shape recovery...")
        
        try:
            # Get viewer to remove and recreate shapes layer
            if self.viewer is not None:
                # Store shape data
                shape_data = []
                try:
                    for shape in self.shapes_layer.data:
                        if isinstance(shape, np.ndarray):
                            # Ensure 3D
                            if shape.ndim == 2 and shape.shape[1] == 2:
                                z_coord = self.current_index
                                z_coords = np.full((len(shape), 1), z_coord, dtype=np.float64)
                                shape_3d = np.hstack([z_coords, shape])
                                shape_data.append(shape_3d)
                            elif shape.ndim == 2 and shape.shape[1] == 3:
                                shape_data.append(shape)
                except:
                    print("Could not read existing shapes")
                
                # Remove the broken layer
                try:
                    self.viewer.layers.remove(self.shapes_layer)
                except:
                    pass
                
                # Create new shapes layer
                self.shapes_layer = None
                self.initialize_shapes_layer()
                
                # Re-add shapes if we saved any
                for shape in shape_data:
                    try:
                        self.shapes_layer.add(shape)
                    except:
                        pass
                
                print("✅ Emergency recovery complete")
                
        except Exception as e:
            print(f"❌ Emergency recovery failed: {e}")


    def _ensure_consistent_dimensions(self) -> None:
        """Ensure all shapes in the layer have consistent dimensions"""
        if self.shapes_layer is None or len(self.shapes_layer.data) == 0:
            return
        
        # Check if we have mixed dimensions
        dimensions = set()
        for shape in self.shapes_layer.data:
            if isinstance(shape, np.ndarray) and shape.size > 0:
                dimensions.add(shape.shape[1] if shape.ndim == 2 else 0)
        
        if len(dimensions) > 1:
            print("⚠️ Mixed dimensions detected, converting all to 3D...")
            try:
                # Temporarily disconnect events
                self.shapes_layer.events.data.disconnect(self._on_shapes_changed)
                
                # Convert all shapes to 3D
                fixed_shapes = []
                for shape in self.shapes_layer.data:
                    if isinstance(shape, np.ndarray):
                        if shape.ndim == 2 and shape.shape[1] == 2:
                            # Convert 2D to 3D
                            z_coord = self.current_index
                            z_coords = np.full((len(shape), 1), z_coord, dtype=np.float64)
                            shape_3d = np.hstack([z_coords, shape])
                            fixed_shapes.append(shape_3d)
                        else:
                            fixed_shapes.append(shape)
                    else:
                        fixed_shapes.append(shape)
                
                # Update all at once
                self.shapes_layer.data = fixed_shapes
                
                # Reconnect
                self.shapes_layer.events.data.connect(self._on_shapes_changed)
                print("✅ Converted all shapes to 3D")
                
            except Exception as e:
                print(f"Error ensuring consistent dimensions: {e}")
                # Reconnect events
                try:
                    self.shapes_layer.events.data.connect(self._on_shapes_changed)
                except:
                    pass


    def _on_shapes_changed(self, event) -> None:
        """Handle changes to shapes layer - track new shapes, modifications, and save annotations"""
        try:
            import time
            print("\n" + "="*80)
            print(f"🔔 SHAPE CHANGE EVENT - DETAILED TRACE @ {time.time():.3f}")
            print("="*80)

            if self.shapes_layer is None:
                print("❌ EARLY EXIT: shapes_layer is None")
                return

            if self._modifying_shapes:
                print("❌ EARLY EXIT: _modifying_shapes flag is True")
                return

            print(f"✅ Passed guard checks")
            print(f"   SAM2 mode: {self.sam2_mode}")
            print(f"   Current object ID: {self.current_sam2_object_id}")
            print(f"   Current ROI params: {self.current_roi_params is not None}")

            # First, ensure all shapes are consistently 3D
            self._ensure_consistent_dimensions()

            current_shapes = self.shapes_layer.data
            current_count = len(current_shapes)
            print(f"   Current shapes count: {current_count}")

            # Get previous state
            previous_shapes = list(self.shape_id_to_coordinates.values())
            previous_count = len(previous_shapes)
            print(f"   Previous shapes count: {previous_count}")

            # Detect modifications BEFORE processing deletions
            modifications = self._detect_shape_modification(previous_shapes, current_shapes)
            print(f"   Detected modifications: {len(modifications) if modifications else 0}")
            
            # If we detected modifications, handle them specially
            if modifications:
                print(f"🔧 Processing {len(modifications)} shape modifications...")
                self._handle_shape_modifications(modifications, current_shapes)
                # Don't process these as new shapes
                return
            
            # Rest of the original _on_shapes_changed logic for deletions and new shapes...
            # First, identify which shapes were removed by comparing coordinates
            removed_shape_ids = []
            for shape_id, stored_coords in list(self.shape_id_to_coordinates.items()):
                found = False
                for current_shape in current_shapes:
                    if self._shapes_match(stored_coords, current_shape):
                        found = True
                        break
                if not found:
                    removed_shape_ids.append(shape_id)
            
            # Handle removed shapes
            if removed_shape_ids:
                print(f"🗑️ Detected removal of shape IDs: {removed_shape_ids}")
                for shape_id in removed_shape_ids:
                    shape_type = self.shape_id_to_type.get(shape_id, 'unknown')
                    obj_id = self.shape_id_to_object.get(shape_id)
                    print(f"  Shape ID {shape_id}: type={shape_type}, object_id={obj_id}")
                    
                    # Remove annotations for this shape ID
                    self._remove_annotations_by_shape_id(shape_id)
                    
                    # Clean up all mappings for this shape ID
                    if shape_id in self.shape_id_to_napari_index:
                        del self.shape_id_to_napari_index[shape_id]
                    if shape_id in self.shape_id_to_type:
                        del self.shape_id_to_type[shape_id]
                    if shape_id in self.shape_id_to_object:
                        del self.shape_id_to_object[shape_id]
                    if shape_id in self.shape_id_to_session:
                        del self.shape_id_to_session[shape_id]
                    if shape_id in self.shape_id_to_coordinates:
                        del self.shape_id_to_coordinates[shape_id]
            
            # Rebuild napari index mapping
            self.napari_index_to_shape_id.clear()
            for napari_idx, current_shape in enumerate(current_shapes):
                # Find if this shape matches any stored shape
                for shape_id, stored_coords in self.shape_id_to_coordinates.items():
                    if self._shapes_match(stored_coords, current_shape):
                        self.napari_index_to_shape_id[napari_idx] = shape_id
                        self.shape_id_to_napari_index[shape_id] = napari_idx
                        break
            
            # Find truly new shapes (not in our shape ID system)
            new_shape_indices = []
            for napari_idx in range(current_count):
                if napari_idx not in self.napari_index_to_shape_id:
                    new_shape_indices.append(napari_idx)
            
            if not new_shape_indices:
                return
            
            print(f"🔍 Processing {len(new_shape_indices)} new shapes: {new_shape_indices}")
            
            # CRITICAL: Check if we're in the middle of adding SAM2 results
            # This prevents SAM2 results from being processed as user ROIs
            adding_sam2_results = getattr(self, '_adding_sam2_results', False)
            
            # Process each new shape
            for napari_idx in sorted(new_shape_indices):
                shape = current_shapes[napari_idx]
                
                # Validate shape
                if not isinstance(shape, np.ndarray) or shape.size == 0:
                    continue
                
                # Assign new shape ID
                shape_id = self._next_shape_id
                self._next_shape_id += 1
                
                # Store mappings
                self.shape_id_to_napari_index[shape_id] = napari_idx
                self.napari_index_to_shape_id[napari_idx] = shape_id
                self.shape_id_to_coordinates[shape_id] = shape.copy()
                
                num_vertices = len(shape)
                print(f"  📝 Shape {napari_idx} (ID: {shape_id}): {num_vertices} vertices")
                
                # ENHANCED: Skip processing if this is a SAM2 result being added
                if adding_sam2_results:
                    print(f"    🤖 SAM2 result detected - skipping auto-processing")
                    continue
                
                # Handle rectangles - check if they should be box prompts
                if len(shape) == 4:
                    print(f"    🔳 Rectangle detected - checking if it's a box prompt")
                    print(f"       SAM2 mode check: {self.sam2_mode} in ['annotation', 'refining']? {self.sam2_mode in ['annotation', 'refining']}")
                    print(f"       ROI params exist: {self.current_roi_params is not None}")

                    # Check if we're in SAM2 annotation mode and this rectangle is inside working ROI
                    if self.sam2_mode in ["annotation", "refining"]:
                        print(f"       ✅ SAM2 mode is correct")

                        if self.current_roi_params is not None:
                            print(f"       ✅ ROI params exist")

                            is_inside = self._is_rectangle_inside_working_roi(shape)
                            print(f"       Rectangle inside working ROI: {is_inside}")

                            if is_inside:
                                print(f"    📦 Rectangle is inside working ROI - converting to box prompt")

                                # Convert to box prompt for current object
                                box_prompt = self._convert_rectangle_to_box_prompt(shape)
                                print(f"       Box prompt result: {box_prompt}")

                                if box_prompt is not None:
                                    obj_id = self.current_sam2_object_id
                                    print(f"       ✅ Storing box prompt for object {obj_id}")

                                    # Store box prompt
                                    self.sam2_box_prompts_by_object[obj_id] = box_prompt
                                    self.object_to_box_shape_index[obj_id] = napari_idx
                                    print(f"       ✅ Box prompt stored successfully!")

                                    # Set shape tracking
                                    self.shape_id_to_object[shape_id] = obj_id
                                    self.shape_id_to_type[shape_id] = 'box_prompt'
                                    self.shape_id_to_session[shape_id] = self.sam2_session_id

                                    # Make object active
                                    self.active_object_ids.add(obj_id)

                                    # Apply box prompt visual style
                                    self._apply_box_prompt_style_to_shape(napari_idx, obj_id)

                                    # Save box prompt annotation (internal SAM2 state, not user annotation)
                                    img_idx = int(shape[0, 0]) if shape.shape[1] == 3 else self.current_index
                                    metadata = {
                                        'shape_id': shape_id,
                                        'box_prompt': box_prompt,
                                        'object_id': obj_id,
                                        'session_id': self.sam2_session_id,
                                        'internal_sam2_state': True  # Mark as internal
                                    }

                                    coords_2d = shape[:, 1:].tolist() if shape.shape[1] == 3 else shape.tolist()

                                    # CHANGED: Don't save box prompts as user annotations
                                    print(f"       ✅ Created box prompt for Object {obj_id}: {box_prompt} (internal state only)")
                                    continue
                                else:
                                    print(f"       ❌ Box prompt conversion returned None")
                            else:
                                print(f"       ❌ Rectangle is NOT inside working ROI")
                        else:
                            print(f"       ❌ ROI params is None")
                    else:
                        print(f"       ❌ SAM2 mode '{self.sam2_mode}' not in ['annotation', 'refining']")
                    
                    # Rectangle not inside working ROI or not in SAM2 mode - store coordinates only
                    print(f"    🔳 Rectangle stored as coordinates only (not a box prompt)")
                    continue
                
                # ENHANCED: For non-rectangles, process as user ROI with edge width 10
                if len(shape) >= 3:
                    self.shape_id_to_object[shape_id] = 0
                    self.shape_id_to_type[shape_id] = 'user_roi'
                    # User ROIs don't get session IDs
                    
                    # ENHANCED: Apply label color with edge width 10
                    self._apply_label_color_to_shape(napari_idx)
                    
                    # Save annotation with shape ID
                    img_idx = int(shape[0, 0]) if shape.shape[1] == 3 else self.current_index
                    
                    # Enhanced save with shape ID
                    metadata = {
                        'shape_id': shape_id,
                        'label': self.selected_label,
                        'label_name': self.label_names[self.selected_label - 1]
                    }
                    
                    # Extract 2D coordinates for saving
                    if shape.shape[1] == 3:
                        coords_2d = shape[:, 1:].tolist()
                    else:
                        coords_2d = shape.tolist()
                    
                    # Create annotation record with shape ID
                    record = AnnotationRecord(
                        timestamp=datetime.now().isoformat(),
                        image_index=img_idx,
                        annotation_type='user_roi',
                        coordinates=coords_2d,
                        object_id=0,
                        metadata=metadata
                    )
                    
                    self.annotation_history.append(record)
                    
                    # Also add to per-image structure
                    if img_idx not in self.all_annotations:
                        self.all_annotations[img_idx] = []
                    
                    self.all_annotations[img_idx].append({
                        'type': 'user_roi',
                        'coordinates': coords_2d,
                        'object_id': 0,
                        'shape_id': shape_id,
                        'label': self.selected_label,
                        'label_name': metadata['label_name'],
                        'metadata': metadata
                    })
                    
                    print(f"💾 Saved user_roi annotation: img={img_idx}, label={self.selected_label}, coords_len={len(coords_2d)} (edge width 10)")
                    
        except Exception as e:
            print(f"❌ Error in _on_shapes_changed: {e}")
            import traceback
            traceback.print_exc()


    def _is_rectangle_inside_working_roi(self, rectangle: np.ndarray) -> bool:
        """Check if a rectangle is inside the current working ROI"""
        if self.current_roi_params is None:
            return False
        
        # Extract rectangle coordinates
        if rectangle.shape[1] == 3:
            y_coords = rectangle[:, 1]
            x_coords = rectangle[:, 2]
        elif rectangle.shape[1] == 2:
            y_coords = rectangle[:, 0]
            x_coords = rectangle[:, 1]
        else:
            return False
        
        # Get bounding box of rectangle
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        
        # Get working ROI bounds
        working_top = self.current_roi_params['top_left_y']
        working_left = self.current_roi_params['top_left_x']
        working_bottom = working_top + self.current_roi_params['height']
        working_right = working_left + self.current_roi_params['width']
        
        # Check if rectangle is entirely inside working ROI
        return (working_top <= min_y < max_y <= working_bottom and
                working_left <= min_x < max_x <= working_right)


    def _convert_rectangle_to_box_prompt(self, rectangle: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """Convert rectangle coordinates to box prompt format (x1, y1, x2, y2) in ROI-local coordinates"""
        if self.current_roi_params is None:
            return None
        
        try:
            # Extract coordinates
            if rectangle.shape[1] == 3:
                y_coords = rectangle[:, 1]
                x_coords = rectangle[:, 2]
            elif rectangle.shape[1] == 2:
                y_coords = rectangle[:, 0]
                x_coords = rectangle[:, 1]
            else:
                return None
            
            # Get global bounding box
            global_x1, global_x2 = np.min(x_coords), np.max(x_coords)
            global_y1, global_y2 = np.min(y_coords), np.max(y_coords)
            
            # Convert to ROI-local coordinates
            local_x1 = global_x1 - self.current_roi_params['top_left_x']
            local_y1 = global_y1 - self.current_roi_params['top_left_y']
            local_x2 = global_x2 - self.current_roi_params['top_left_x']
            local_y2 = global_y2 - self.current_roi_params['top_left_y']
            
            # Ensure coordinates are within ROI bounds
            roi_width = self.current_roi_params['width']
            roi_height = self.current_roi_params['height']
            
            local_x1 = max(0, min(local_x1, roi_width))
            local_y1 = max(0, min(local_y1, roi_height))
            local_x2 = max(0, min(local_x2, roi_width))
            local_y2 = max(0, min(local_y2, roi_height))
            
            # Return in SAM2 format: (x1, y1, x2, y2)
            return (float(local_x1), float(local_y1), float(local_x2), float(local_y2))
            
        except Exception as e:
            print(f"Error converting rectangle to box prompt: {e}")
            return None


    def _apply_box_prompt_style_to_shape(self, shape_idx: int, object_id: int) -> None:
        """Apply box prompt visual style to a shape"""
        try:
            if self.shapes_layer is None or shape_idx >= len(self.shapes_layer.data):
                return
            
            # Box prompt style - dashed border with object color
            object_colors = ['cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray']
            color_idx = (object_id - 1) % len(object_colors)
            box_color = object_colors[color_idx]
            
            # Apply style
            if hasattr(self.shapes_layer, 'edge_color'):
                colors = list(self.shapes_layer.edge_color)
                while len(colors) <= shape_idx:
                    colors.append('yellow')
                colors[shape_idx] = box_color
                self.shapes_layer.edge_color = colors
            
            if hasattr(self.shapes_layer, 'edge_width'):
                widths = list(self.shapes_layer.edge_width)
                while len(widths) <= shape_idx:
                    widths.append(3)
                widths[shape_idx] = 3
                self.shapes_layer.edge_width = widths
            
            print(f"🎨 Applied box prompt style to shape {shape_idx} (Object {object_id}, {box_color})")
            
        except Exception as e:
            print(f"Warning: Could not apply box prompt style to shape {shape_idx}: {e}")


    def debug_shape_state(self) -> None:
        """Print detailed state of all shapes and their mappings"""
        print("\n" + "="*60)
        print("SHAPE STATE DEBUG")
        print("="*60)
        
        if self.shapes_layer is None:
            print("No shapes layer")
            return
        
        print(f"Total shapes in napari layer: {len(self.shapes_layer.data)}")
        print(f"Next shape ID: {self._next_shape_id}")
        
        print("\nShape ID -> Coordinates mapping:")
        for shape_id, coords in sorted(self.shape_id_to_coordinates.items()):
            print(f"  Shape ID {shape_id}: {len(coords)} vertices")
        
        print("\nShape ID -> Type mapping:")
        for shape_id, shape_type in sorted(self.shape_id_to_type.items()):
            obj_id = self.shape_id_to_object.get(shape_id)
            session = self.shape_id_to_session.get(shape_id)
            print(f"  Shape ID {shape_id}: type={shape_type}, object={obj_id}, session={session}")
        
        print("\nNapari index -> Shape ID mapping:")
        for idx in range(len(self.shapes_layer.data)):
            shape_id = self.napari_index_to_shape_id.get(idx)
            shape = self.shapes_layer.data[idx]
            if shape_id is not None:
                shape_type = self.shape_id_to_type.get(shape_id, 'NO TYPE')
                print(f"  [{idx}] -> Shape ID {shape_id} ({len(shape)} vertices, type={shape_type})")
            else:
                print(f"  [{idx}] -> NO SHAPE ID ({len(shape)} vertices)")
        
        print("\nOrphaned shape IDs (have coordinates but no napari index):")
        for shape_id in self.shape_id_to_coordinates.keys():
            if shape_id not in self.shape_id_to_napari_index:
                print(f"  Shape ID {shape_id}")
        
        print("="*60)


    def _handle_shape_modifications(self, modifications: Dict[int, int], current_shapes: List) -> None:
        """Handle shapes that were modified (not deleted or added)"""
        
        self._modifying_shapes = True  # Prevent recursion
        
        try:
            successfully_updated = []
            failed_updates = []
            
            for old_idx, new_idx in modifications.items():
                if old_idx not in self.shape_selection_snapshot['shapes_data']:
                    print(f"⚠️ No snapshot data for modified shape at index {old_idx}")
                    failed_updates.append((old_idx, new_idx))
                    continue
                
                snapshot_data = self.shape_selection_snapshot['shapes_data'][old_idx]
                shape_id = snapshot_data['shape_id']
                metadata = snapshot_data['metadata']
                
                if shape_id is None:
                    print(f"⚠️ No shape ID in snapshot for index {old_idx}")
                    failed_updates.append((old_idx, new_idx))
                    continue
                
                # Update the shape coordinates in our tracking
                new_shape_coords = current_shapes[new_idx]
                self.shape_id_to_coordinates[shape_id] = new_shape_coords.copy()
                
                # Update index mappings if needed
                if old_idx != new_idx:
                    self.shape_id_to_napari_index[shape_id] = new_idx
                    self.napari_index_to_shape_id[new_idx] = shape_id
                    if old_idx in self.napari_index_to_shape_id:
                        del self.napari_index_to_shape_id[old_idx]
                
                # Update the annotation records with new coordinates
                self._update_annotation_coordinates(shape_id, new_shape_coords, metadata)
                
                shape_type = metadata.get('type', 'unknown')
                print(f"✅ Updated {shape_type} shape {shape_id}: index {old_idx}→{new_idx}")
                successfully_updated.append((old_idx, shape_id))
            
            # Handle failed updates
            if failed_updates:
                print(f"\n⚠️ WARNING: {len(failed_updates)} shape modifications could not be tracked correctly!")
                print("The changes in these ROIs were not correctly tracked.")
                
                # Revert failed modifications
                for old_idx, new_idx in failed_updates:
                    if old_idx in self.shape_selection_snapshot['shapes_data']:
                        old_coords = self.shape_selection_snapshot['shapes_data'][old_idx]['coordinates']
                        
                        # Restore original shape
                        try:
                            # Temporarily disconnect events
                            self.shapes_layer.events.data.disconnect(self._on_shapes_changed)
                            
                            # Restore the original coordinates
                            shapes_list = list(self.shapes_layer.data)
                            shapes_list[new_idx] = old_coords
                            self.shapes_layer.data = shapes_list
                            
                            print(f"🔄 Reverted shape at index {new_idx} to original coordinates")
                            
                            # Reconnect events
                            self.shapes_layer.events.data.connect(self._on_shapes_changed)
                            
                        except Exception as e:
                            print(f"❌ Could not revert shape at index {new_idx}: {e}")
                            # Ensure events are reconnected
                            try:
                                self.shapes_layer.events.data.connect(self._on_shapes_changed)
                            except:
                                pass
            
            # Clear the snapshot after processing
            self.shape_selection_snapshot = {
                'selected_indices': set(),
                'shapes_data': {},
                'snapshot_time': None
            }
            
        finally:
            self._modifying_shapes = False


    def _update_annotation_coordinates(self, shape_id: int, new_coords: np.ndarray, metadata: Dict) -> None:
        """Update annotation coordinates while preserving all metadata"""
        
        # Extract 2D coordinates for storage
        if new_coords.shape[1] == 3:
            coords_2d = new_coords[:, 1:].tolist()  # Remove Z coordinate
            img_idx = int(new_coords[0, 0])
        else:
            coords_2d = new_coords.tolist()
            img_idx = self.current_index
        
        # Update in annotation history
        updated_in_history = False
        for record in self.annotation_history:
            if record.metadata and record.metadata.get('shape_id') == shape_id:
                record.coordinates = coords_2d
                updated_in_history = True
                print(f"  📝 Updated coordinates in annotation history")
                break
        
        # Update in per-image annotations
        if img_idx in self.all_annotations:
            for ann in self.all_annotations[img_idx]:
                if ann.get('shape_id') == shape_id or ann.get('metadata', {}).get('shape_id') == shape_id:
                    ann['coordinates'] = coords_2d
                    print(f"  📝 Updated coordinates in image {img_idx} annotations")
                    break
        
        if not updated_in_history:
            print(f"  ⚠️ Warning: Could not find annotation record for shape ID {shape_id}")


    def _handle_removed_shapes(self, removed_indices: Set[int]) -> None:
        """Handle shapes that were removed from the layer - clean up annotations"""
        print(f"🗑️ Handling removal of shapes: {removed_indices}")
        
        removed_annotations_count = 0
        
        for idx in removed_indices:
            obj_id = self.shape_to_object_mapping.get(idx)
            shape_type = self.shape_to_type_mapping.get(idx)
            
            print(f"  Shape {idx}: type={shape_type}, object_id={obj_id}")
            
            # Remove from annotations based on shape type and object ID
            if shape_type and obj_id is not None:
                # Find and remove corresponding annotations
                original_history_count = len(self.annotation_history)
                
                # Remove from annotation history
                if shape_type == 'sam2_result' and obj_id > 0:
                    # Remove SAM2 result annotations for this object
                    self.annotation_history = [
                        record for record in self.annotation_history 
                        if not (record.annotation_type == 'sam2_result' and record.object_id == obj_id)
                    ]
                elif shape_type == 'user_roi':
                    # Remove user ROI annotations (need to match by coordinates or timestamp)
                    # This is trickier since we don't have direct object mapping
                    # For now, remove the most recent user_roi annotation
                    for i in range(len(self.annotation_history) - 1, -1, -1):
                        if self.annotation_history[i].annotation_type == 'user_roi':
                            del self.annotation_history[i]
                            break
                elif shape_type in ['working_roi', 'box_prompt']:
                    # Remove working ROI and box prompt annotations
                    self.annotation_history = [
                        record for record in self.annotation_history 
                        if not (record.annotation_type == shape_type and record.object_id == obj_id)
                    ]
                
                removed_from_history = original_history_count - len(self.annotation_history)
                
                # Remove from per-image annotations
                removed_from_images = 0
                for img_idx in list(self.all_annotations.keys()):
                    original_img_count = len(self.all_annotations[img_idx])
                    
                    if shape_type == 'sam2_result' and obj_id > 0:
                        self.all_annotations[img_idx] = [
                            ann for ann in self.all_annotations[img_idx]
                            if not (ann.get('type') == 'sam2_result' and ann.get('object_id') == obj_id)
                        ]
                    elif shape_type == 'user_roi':
                        # Remove one user ROI annotation (most recent)
                        for i in range(len(self.all_annotations[img_idx]) - 1, -1, -1):
                            if self.all_annotations[img_idx][i].get('type') == 'user_roi':
                                del self.all_annotations[img_idx][i]
                                break
                    elif shape_type in ['working_roi', 'box_prompt']:
                        self.all_annotations[img_idx] = [
                            ann for ann in self.all_annotations[img_idx]
                            if not (ann.get('type') == shape_type and ann.get('object_id') == obj_id)
                        ]
                    
                    removed_from_images += original_img_count - len(self.all_annotations[img_idx])
                    
                    # Remove empty image entries
                    if not self.all_annotations[img_idx]:
                        del self.all_annotations[img_idx]
                
                if removed_from_history > 0 or removed_from_images > 0:
                    print(f"    Removed {removed_from_history} from history, {removed_from_images} from image records")
                    removed_annotations_count += removed_from_history
            
            # Clean up tracking
            if idx in self.shape_to_object_mapping:
                del self.shape_to_object_mapping[idx]
            if idx in self.shape_to_type_mapping:
                del self.shape_to_type_mapping[idx]
            if idx in self.shape_to_session_mapping:
                del self.shape_to_session_mapping[idx]
                
            if shape_type == 'working_roi':
                self.sam2_working_roi_id = None
            elif shape_type == 'box_prompt' and obj_id is not None:
                if obj_id in self.sam2_box_prompts_by_object:
                    del self.sam2_box_prompts_by_object[obj_id]
                if obj_id in self.object_to_box_shape_index:
                    del self.object_to_box_shape_index[obj_id]
                    
        self._reindex_shape_mappings(removed_indices)
        
        if removed_annotations_count > 0:
            print(f"✅ Removed {removed_annotations_count} annotations from manual shape deletion")
        else:
            print("ℹ️ No annotations removed (shapes may not have been saved yet)")
    

    def _handle_removed_shapes_v2(self, removed_napari_indices: Set[int]) -> None:
        """Handle shapes that were removed using shape IDs"""
        print(f"🗑️ Handling removal of shapes at napari indices: {removed_napari_indices}")
        
        # Find shape IDs for removed indices
        removed_shape_ids = []
        for napari_idx in removed_napari_indices:
            if napari_idx in self.napari_index_to_shape_id:
                shape_id = self.napari_index_to_shape_id[napari_idx]
                removed_shape_ids.append(shape_id)
                
                shape_type = self.shape_id_to_type.get(shape_id, 'unknown')
                obj_id = self.shape_id_to_object.get(shape_id)
                print(f"  Shape ID {shape_id} (napari idx {napari_idx}): type={shape_type}, object_id={obj_id}")
        
        # Remove annotations by shape ID
        for shape_id in removed_shape_ids:
            self._remove_annotations_by_shape_id(shape_id)
            
            # Clean up shape ID mappings
            if shape_id in self.shape_id_to_napari_index:
                del self.shape_id_to_napari_index[shape_id]
            if shape_id in self.shape_id_to_type:
                del self.shape_id_to_type[shape_id]
            if shape_id in self.shape_id_to_object:
                del self.shape_id_to_object[shape_id]
            if shape_id in self.shape_id_to_session:
                del self.shape_id_to_session[shape_id]
            if shape_id in self.shape_id_to_coordinates:
                del self.shape_id_to_coordinates[shape_id]


    def _reindex_shape_mappings(self, removed_indices: Set[int]) -> None:
        """Reindex shape mappings after shapes are removed"""
        if not removed_indices:
            return
            
        sorted_removed = sorted(removed_indices, reverse=True)
        
        new_shape_to_object = {}
        new_shape_to_type = {}
        new_object_to_box = {}
        
        for old_idx, obj_id in self.shape_to_object_mapping.items():
            if old_idx in removed_indices:
                continue
                
            new_idx = old_idx
            for removed_idx in sorted_removed:
                if old_idx > removed_idx:
                    new_idx -= 1
                    
            new_shape_to_object[new_idx] = obj_id
            new_shape_to_type[new_idx] = self.shape_to_type_mapping.get(old_idx, '')
            
            if self.shape_to_type_mapping.get(old_idx) == 'box_prompt' and obj_id > 0:
                new_object_to_box[obj_id] = new_idx
                
        if self.sam2_working_roi_id is not None:
            new_working_roi_id = self.sam2_working_roi_id
            for removed_idx in sorted_removed:
                if self.sam2_working_roi_id > removed_idx:
                    new_working_roi_id -= 1
            self.sam2_working_roi_id = new_working_roi_id if new_working_roi_id >= 0 else None
            
        self.shape_to_object_mapping = new_shape_to_object
        self.shape_to_type_mapping = new_shape_to_type
        self.object_to_box_shape_index = new_object_to_box
    

    def _rebuild_napari_index_mapping(self) -> None:
        """Rebuild the napari index to shape ID mapping after changes"""
        if self.shapes_layer is None:
            return
            
        # Clear old napari index mapping
        self.napari_index_to_shape_id.clear()
        
        # Rebuild based on coordinate matching
        current_shapes = self.shapes_layer.data
        
        for shape_id, stored_coords in self.shape_id_to_coordinates.items():
            # Find this shape in current data
            for napari_idx, current_shape in enumerate(current_shapes):
                if self._shapes_match(stored_coords, current_shape):
                    self.napari_index_to_shape_id[napari_idx] = shape_id
                    self.shape_id_to_napari_index[shape_id] = napari_idx
                    break
                    

    def _on_positive_points_changed(self) -> None:
        """Handle when positive points are added"""
        try:
            if (self.positive_points_layer is None or 
                self.current_roi_params is None or 
                self.sam2_mode not in ["annotation", "refining"]):
                return
            
            current_points = self.positive_points_layer.data
            current_count = len(current_points)
            
            new_point_indices = set(range(current_count)) - self.processed_positive_points
            
            if not new_point_indices:
                if len(self.processed_positive_points) > current_count:
                    self._handle_removed_positive_points(current_count)
                return
                
            for point_idx in sorted(new_point_indices):
                point = current_points[point_idx]
                
                if len(point) == 3:
                    image_idx = int(point[0])
                    global_coords = (float(point[1]), float(point[2]))
                else:
                    image_idx = self.current_index
                    global_coords = (float(point[0]), float(point[1]))
                
                local_coords = self.global_to_local_coords(global_coords, self.current_roi_params)
                roi_height = self.current_roi_params['height']
                roi_width = self.current_roi_params['width']
                
                if (0 <= local_coords[0] <= roi_height and 0 <= local_coords[1] <= roi_width):
                    annotation = PointAnnotation(
                        coordinates=global_coords,
                        point_type='positive',
                        roi_local_coords=local_coords,
                        image_index=image_idx
                    )
                    
                    obj_id = self.current_sam2_object_id
                    if obj_id not in self.point_annotations_by_object:
                        self.point_annotations_by_object[obj_id] = []
                        
                    self.point_annotations_by_object[obj_id].append(annotation)
                    self.positive_point_to_object[point_idx] = obj_id
                    self.active_object_ids.add(obj_id)
                    
                    # REMOVED: Point annotation saving - points are internal SAM2 state only
                    print(f"✅ Positive point assigned to Object {obj_id} (internal state only)")
                else:
                    print(f"⚠️ Positive point is outside ROI bounds")
                    
                self.processed_positive_points.add(point_idx)
                
        except Exception as e:
            print(f"❌ Error processing positive points: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_negative_points_changed(self) -> None:
        """Handle when negative points are added"""
        try:
            if (self.negative_points_layer is None or 
                self.current_roi_params is None or 
                self.sam2_mode not in ["annotation", "refining"]):
                return
            
            current_points = self.negative_points_layer.data
            current_count = len(current_points)
            
            new_point_indices = set(range(current_count)) - self.processed_negative_points
            
            if not new_point_indices:
                if len(self.processed_negative_points) > current_count:
                    self._handle_removed_negative_points(current_count)
                return
                
            for point_idx in sorted(new_point_indices):
                point = current_points[point_idx]
                
                if len(point) == 3:
                    image_idx = int(point[0])
                    global_coords = (float(point[1]), float(point[2]))
                else:
                    image_idx = self.current_index
                    global_coords = (float(point[0]), float(point[1]))
                
                local_coords = self.global_to_local_coords(global_coords, self.current_roi_params)
                roi_height = self.current_roi_params['height']
                roi_width = self.current_roi_params['width']
                
                if (0 <= local_coords[0] <= roi_height and 0 <= local_coords[1] <= roi_width):
                    annotation = PointAnnotation(
                        coordinates=global_coords,
                        point_type='negative',
                        roi_local_coords=local_coords,
                        image_index=image_idx
                    )
                    
                    obj_id = self.current_sam2_object_id
                    if obj_id not in self.point_annotations_by_object:
                        self.point_annotations_by_object[obj_id] = []
                        
                    self.point_annotations_by_object[obj_id].append(annotation)
                    self.negative_point_to_object[point_idx] = obj_id
                    self.active_object_ids.add(obj_id)
                    
                    # REMOVED: Point annotation saving - points are internal SAM2 state only
                    print(f"✅ Negative point assigned to Object {obj_id} (internal state only)")
                else:
                    print(f"⚠️ Negative point is outside ROI bounds")
                    
                self.processed_negative_points.add(point_idx)
                
        except Exception as e:
            print(f"❌ Error processing negative points: {e}")
            import traceback
            traceback.print_exc()
    

    def _handle_removed_positive_points(self, new_count: int) -> None:
        """Handle when positive points are removed"""
        removed_indices = self.processed_positive_points - set(range(new_count))
        
        for idx in removed_indices:
            obj_id = self.positive_point_to_object.get(idx)
            if obj_id is not None:
                print(f"⚠️ Positive point {idx} removed (was assigned to Object {obj_id})")
                del self.positive_point_to_object[idx]
                
        self.processed_positive_points = self.processed_positive_points - removed_indices
        self._reindex_point_mappings('positive', removed_indices)
    
    def _handle_removed_negative_points(self, new_count: int) -> None:
        """Handle when negative points are removed"""
        removed_indices = self.processed_negative_points - set(range(new_count))
        
        for idx in removed_indices:
            obj_id = self.negative_point_to_object.get(idx)
            if obj_id is not None:
                print(f"⚠️ Negative point {idx} removed (was assigned to Object {obj_id})")
                del self.negative_point_to_object[idx]
                
        self.processed_negative_points = self.processed_negative_points - removed_indices
        self._reindex_point_mappings('negative', removed_indices)
    
    def _reindex_point_mappings(self, point_type: str, removed_indices: Set[int]) -> None:
        """Reindex point mappings after points are removed"""
        if not removed_indices:
            return
            
        sorted_removed = sorted(removed_indices, reverse=True)
        
        if point_type == 'positive':
            old_mapping = self.positive_point_to_object.copy()
            new_mapping = {}
            new_processed = set()
            
            for old_idx, obj_id in old_mapping.items():
                if old_idx in removed_indices:
                    continue
                    
                new_idx = old_idx
                for removed_idx in sorted_removed:
                    if old_idx > removed_idx:
                        new_idx -= 1
                        
                new_mapping[new_idx] = obj_id
                new_processed.add(new_idx)
                
            self.positive_point_to_object = new_mapping
            self.processed_positive_points = new_processed
            
        else:
            old_mapping = self.negative_point_to_object.copy()
            new_mapping = {}
            new_processed = set()
            
            for old_idx, obj_id in old_mapping.items():
                if old_idx in removed_indices:
                    continue
                    
                new_idx = old_idx
                for removed_idx in sorted_removed:
                    if old_idx > removed_idx:
                        new_idx -= 1
                        
                new_mapping[new_idx] = obj_id
                new_processed.add(new_idx)
                
            self.negative_point_to_object = new_mapping
            self.processed_negative_points = new_processed
    
    def validate_annotation_state(self) -> bool:
        """Validate that all annotation mappings are consistent"""
        errors = []
        
        if self.shapes_layer is not None:
            current_shape_count = len(self.shapes_layer.data)
            for shape_idx in self.shape_to_object_mapping.keys():
                if shape_idx >= current_shape_count:
                    errors.append(f"Shape index {shape_idx} in mapping but only {current_shape_count} shapes exist")
                    
        if self.positive_points_layer is not None:
            pos_count = len(self.positive_points_layer.data)
            for point_idx in self.positive_point_to_object.keys():
                if point_idx >= pos_count:
                    errors.append(f"Positive point index {point_idx} in mapping but only {pos_count} points exist")
                    
        if self.negative_points_layer is not None:
            neg_count = len(self.negative_points_layer.data)
            for point_idx in self.negative_point_to_object.keys():
                if point_idx >= neg_count:
                    errors.append(f"Negative point index {point_idx} in mapping but only {neg_count} points exist")
                    
        for obj_id in self.active_object_ids:
            has_points = obj_id in self.point_annotations_by_object and len(self.point_annotations_by_object[obj_id]) > 0
            has_box = obj_id in self.sam2_box_prompts_by_object and self.sam2_box_prompts_by_object[obj_id] is not None
            
            if not has_points and not has_box:
                errors.append(f"Object {obj_id} is active but has no annotations")
                
        if self.sam2_working_roi_id is not None:
            if self.shapes_layer is None or self.sam2_working_roi_id >= len(self.shapes_layer.data):
                errors.append(f"Working ROI index {self.sam2_working_roi_id} is invalid")
                
        if errors:
            print("❌ Annotation state validation failed:")
            for error in errors:
                print(f"   - {error}")
            return False
        else:
            print("✅ Annotation state is valid")
            return True
    
    def clear_annotations_for_object(self, object_id: int) -> None:
        """Clear all annotations for a specific object"""
        if object_id in self.point_annotations_by_object:
            del self.point_annotations_by_object[object_id]
            
        if object_id in self.sam2_box_prompts_by_object:
            del self.sam2_box_prompts_by_object[object_id]
            
        self.active_object_ids.discard(object_id)
        
        if object_id in self.object_to_box_shape_index:
            del self.object_to_box_shape_index[object_id]
            
        print(f"✅ Cleared all annotations for Object {object_id}")
    
    def get_object_summary(self, object_id: int) -> str:
        """Get a summary of annotations for a specific object"""
        summary_parts = [f"Object {object_id}:"]
        
        if object_id in self.point_annotations_by_object:
            annotations = self.point_annotations_by_object[object_id]
            pos_count = sum(1 for a in annotations if a.point_type == 'positive')
            neg_count = sum(1 for a in annotations if a.point_type == 'negative')
            
            points_per_image = {}
            for ann in annotations:
                if ann.image_index not in points_per_image:
                    points_per_image[ann.image_index] = {'positive': 0, 'negative': 0}
                points_per_image[ann.image_index][ann.point_type] += 1
                
            summary_parts.append(f"  Points: {pos_count} positive, {neg_count} negative")
            for img_idx, counts in sorted(points_per_image.items()):
                summary_parts.append(f"    Image {img_idx}: {counts['positive']}+, {counts['negative']}-")
        else:
            summary_parts.append("  Points: None")
            
        # ENHANCED: Better box prompt display
        if object_id in self.sam2_box_prompts_by_object and self.sam2_box_prompts_by_object[object_id]:
            box = self.sam2_box_prompts_by_object[object_id]
            summary_parts.append(f"  Box prompt: ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}) [ROI-local coords]")
            
            # Also show if there's a shape index for the box
            if object_id in self.object_to_box_shape_index:
                shape_idx = self.object_to_box_shape_index[object_id]
                summary_parts.append(f"    Box shape index: {shape_idx}")
        else:
            summary_parts.append("  Box prompt: None")
            
        if object_id in self.active_object_ids:
            summary_parts.append("  Status: ✅ Active")
        else:
            summary_parts.append("  Status: ❌ Inactive")
            
        return "\n".join(summary_parts)
    
    def clear_point_layers(self) -> None:
        """Clear all points from both positive and negative layers"""
        try:
            if self.positive_points_layer is not None:
                ndim = 3
                try:
                    self.positive_points_layer.data = np.empty((0, ndim))
                except Exception as e:
                    print(f"Warning: Could not clear positive points: {e}")
                    # Try alternative clearing method
                    try:
                        self.positive_points_layer.selected_data = set(range(len(self.positive_points_layer.data)))
                        self.positive_points_layer.remove_selected()
                    except:
                        pass
                self._last_positive_count = 0
                    
            if self.negative_points_layer is not None:
                ndim = 3
                try:
                    self.negative_points_layer.data = np.empty((0, ndim))
                except Exception as e:
                    print(f"Warning: Could not clear negative points: {e}")
                    # Try alternative clearing method
                    try:
                        self.negative_points_layer.selected_data = set(range(len(self.negative_points_layer.data)))
                        self.negative_points_layer.remove_selected()
                    except:
                        pass
                self._last_negative_count = 0
            
            self.point_annotations_by_object.clear()
            self.positive_point_to_object.clear()
            self.negative_point_to_object.clear()
            self.processed_positive_points.clear()
            self.processed_negative_points.clear()
            
            self._recompute_active_objects()
            
            print("✅ Point layers cleared")
        except Exception as e:
            # synchronize mappings
            self.synchronize_all_mappings()

            print(f"❌ Warning: Could not clear point layers completely: {e}")
            # Continue with cleanup even if point clearing fails
            self.point_annotations_by_object.clear()
            self.positive_point_to_object.clear()
            self.negative_point_to_object.clear()
            self.processed_positive_points.clear()
            self.processed_negative_points.clear()
            self._recompute_active_objects()
    
    def _recompute_active_objects(self) -> None:
        """Recompute which objects are active based on current annotations"""
        new_active = set()
        
        for obj_id, annotations in self.point_annotations_by_object.items():
            if annotations:
                new_active.add(obj_id)
        
        for obj_id, box_prompt in self.sam2_box_prompts_by_object.items():
            if box_prompt is not None:
                new_active.add(obj_id)
        
        self.active_object_ids = new_active
        print(f"✅ Recomputed active objects: {sorted(self.active_object_ids)}")
    

    def _apply_label_color_to_shape(self, shape_idx: int) -> None:
        """Apply the selected label's color to a shape with edge width 10"""
        try:
            if self.shapes_layer is None or shape_idx >= len(self.shapes_layer.data):
                return
            
            # FLEXIBLE: Validate label is within range
            if not (1 <= self.selected_label <= self.num_labels):
                print(f"Warning: Invalid label {self.selected_label}")
                return
            
            # Convert colormap color to napari-compatible format
            label_color_rgb = self.colormap[self.selected_label - 1]  # Get RGB values (0-1 range)
            
            # Convert to hex color for napari
            label_color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(label_color_rgb[0] * 255),
                int(label_color_rgb[1] * 255), 
                int(label_color_rgb[2] * 255)
            )
            
            # Apply color and ENHANCED edge width to the shape
            if hasattr(self.shapes_layer, 'edge_color'):
                colors = list(self.shapes_layer.edge_color)
                while len(colors) <= shape_idx:
                    colors.append('yellow')  # Default color
                colors[shape_idx] = label_color_hex
                self.shapes_layer.edge_color = colors
            
            # ENHANCED: Set edge width to 10 for finalized/labeled ROIs
            if hasattr(self.shapes_layer, 'edge_width'):
                widths = list(self.shapes_layer.edge_width)
                while len(widths) <= shape_idx:
                    widths.append(3)  # Default width
                widths[shape_idx] = 10  # Enhanced visibility for finalized ROIs
                self.shapes_layer.edge_width = widths
            
            print(f"🎨 Applied label {self.selected_label} color ({self.label_names[self.selected_label - 1]}) and edge width 10 to shape {shape_idx}")
            
        except Exception as e:
            print(f"Warning: Could not apply color and edge width to shape {shape_idx}: {e}")

    
    def convert_rectangle_to_user_roi(self, shape_idx: int) -> None:
        """Convert an untracked rectangle to a user ROI with current label"""
        try:
            # synchronize mappings
            self.synchronize_all_mappings()

            if (self.shapes_layer is None or 
                shape_idx >= len(self.shapes_layer.data) or
                shape_idx in self.shape_to_object_mapping):
                return
            
            shape = self.shapes_layer.data[shape_idx]
            
            # Only process rectangles
            if len(shape) != 4:
                return
            
            # Mark as user ROI
            self.shape_to_object_mapping[shape_idx] = 0
            self.shape_to_type_mapping[shape_idx] = 'user_roi'
            
            # Apply color
            self._apply_label_color_to_shape(shape_idx)
            
            # Save annotation - extract Y,X coordinates
            shape_coords = shape[:, 1:] if shape.shape[1] == 3 else shape
            img_idx = int(shape[0, 0]) if shape.shape[1] == 3 else self.current_index
            
            self.save_annotation(
                img_idx,
                'user_roi', 
                shape_coords,
                label=self.selected_label
            )
            
            print(f"✅ Converted rectangle {shape_idx} to user ROI with label {self.selected_label} ({self.label_names[self.selected_label - 1]})")
            
        except Exception as e:
            # synchronize mappings
            self.synchronize_all_mappings()

            print(f"Error converting rectangle to user ROI: {e}")
    
    def ensure_all_shapes_3d(self) -> None:
        """Ensure all shapes in the layer are 3D to prevent dimensionality issues"""
        if self.shapes_layer is None or len(self.shapes_layer.data) == 0:
            return
        
        try:
            print("🔧 Checking shape dimensionality consistency...")
            shapes_data = self.shapes_layer.data
            needs_fixing = False
            
            # Check if any shapes are 2D
            for i, shape in enumerate(shapes_data):
                if isinstance(shape, np.ndarray):
                    if shape.ndim == 2 and shape.shape[1] == 2:
                        needs_fixing = True
                        print(f"Found 2D shape at index {i}")
                        break
            
            if needs_fixing:
                print("⚠️ Found mixed 2D/3D shapes - converting all to 3D...")
                
                # Temporarily disconnect events
                try:
                    self.shapes_layer.events.data.disconnect(self._on_shapes_changed)
                except:
                    pass
                
                # Clear the layer and re-add all shapes as 3D
                fixed_shapes = []
                fixed_properties = {
                    'edge_color': [],
                    'edge_width': [],
                    'face_color': []
                }
                
                for i, shape in enumerate(shapes_data):
                    if isinstance(shape, np.ndarray):
                        shape_array = np.array(shape)
                        
                        if shape_array.ndim == 2:
                            if shape_array.shape[1] == 2:
                                # Convert 2D to 3D using current image index
                                z_coords = np.full((len(shape_array), 1), self.current_index, dtype=np.float64)
                                shape_3d = np.hstack([z_coords, shape_array])
                                fixed_shapes.append(shape_3d)
                            elif shape_array.shape[1] == 3:
                                # Already 3D
                                fixed_shapes.append(shape_array)
                            else:
                                print(f"Skipping shape {i} with invalid dimensions: {shape_array.shape}")
                                continue
                        else:
                            print(f"Skipping shape {i} with invalid structure")
                            continue
                    else:
                        print(f"Skipping non-array shape {i}")
                        continue
                    
                    # Preserve visual properties
                    try:
                        if hasattr(self.shapes_layer, 'edge_color') and i < len(self.shapes_layer.edge_color):
                            fixed_properties['edge_color'].append(self.shapes_layer.edge_color[i])
                        else:
                            fixed_properties['edge_color'].append('yellow')
                            
                        if hasattr(self.shapes_layer, 'edge_width') and i < len(self.shapes_layer.edge_width):
                            fixed_properties['edge_width'].append(self.shapes_layer.edge_width[i])
                        else:
                            fixed_properties['edge_width'].append(3)
                            
                        if hasattr(self.shapes_layer, 'face_color') and i < len(self.shapes_layer.face_color):
                            fixed_properties['face_color'].append(self.shapes_layer.face_color[i])
                        else:
                            fixed_properties['face_color'].append([0, 0, 0, 0])
                    except:
                        fixed_properties['edge_color'].append('yellow')
                        fixed_properties['edge_width'].append(3)
                        fixed_properties['face_color'].append([0, 0, 0, 0])
                
                # Clear and recreate the layer with fixed shapes
                if fixed_shapes:
                    self.shapes_layer.data = fixed_shapes
                    try:
                        self.shapes_layer.edge_color = fixed_properties['edge_color']
                        self.shapes_layer.edge_width = fixed_properties['edge_width']
                        self.shapes_layer.face_color = fixed_properties['face_color']
                    except Exception as e:
                        # synchronize mappings
                        self.synchronize_all_mappings()

                        print(f"Warning: Could not restore shape properties: {e}")
                    
                    print(f"✅ Fixed {len(fixed_shapes)} shapes to be consistently 3D")
                else:
                    print("⚠️ No valid shapes found after conversion")
                
                # Reconnect events
                self.shapes_layer.events.data.connect(self._on_shapes_changed)
            else:
                print("✅ All shapes are already 3D")
                
        except Exception as e:
            # synchronize mappings
            self.synchronize_all_mappings()

            print(f"❌ Error ensuring 3D shapes: {e}")
            import traceback
            traceback.print_exc()
            
            # Make sure events are reconnected
            try:
                self.shapes_layer.events.data.connect(self._on_shapes_changed)
            except:
                pass
        """Ensure all shapes in the layer are 3D to prevent dimensionality issues"""
        if self.shapes_layer is None or len(self.shapes_layer.data) == 0:
            return
        
        try:
            print("🔧 Checking shape dimensionality consistency...")
            shapes_data = self.shapes_layer.data
            needs_fixing = False
            
            # Check if any shapes are 2D
            for i, shape in enumerate(shapes_data):
                if isinstance(shape, np.ndarray):
                    if shape.ndim == 2 and shape.shape[1] == 2:
                        needs_fixing = True
                        print(f"Found 2D shape at index {i}")
                        break
            
            if needs_fixing:
                print("⚠️ Found mixed 2D/3D shapes - converting all to 3D...")
                
                # Temporarily disconnect events
                try:
                    self.shapes_layer.events.data.disconnect(self._on_shapes_changed)
                except:
                    pass
                
                # Clear the layer and re-add all shapes as 3D
                fixed_shapes = []
                fixed_properties = {
                    'edge_color': [],
                    'edge_width': [],
                    'face_color': []
                }
                
                for i, shape in enumerate(shapes_data):
                    if isinstance(shape, np.ndarray):
                        shape_array = np.array(shape)
                        
                        if shape_array.ndim == 2:
                            if shape_array.shape[1] == 2:
                                # Convert 2D to 3D using current image index
                                z_coords = np.full((len(shape_array), 1), self.current_index, dtype=np.float64)
                                shape_3d = np.hstack([z_coords, shape_array])
                                fixed_shapes.append(shape_3d)
                            elif shape_array.shape[1] == 3:
                                # Already 3D
                                fixed_shapes.append(shape_array)
                            else:
                                print(f"Skipping shape {i} with invalid dimensions: {shape_array.shape}")
                                continue
                        else:
                            print(f"Skipping shape {i} with invalid structure")
                            continue
                    else:
                        print(f"Skipping non-array shape {i}")
                        continue
                    
                    # Preserve visual properties
                    try:
                        if hasattr(self.shapes_layer, 'edge_color') and i < len(self.shapes_layer.edge_color):
                            fixed_properties['edge_color'].append(self.shapes_layer.edge_color[i])
                        else:
                            fixed_properties['edge_color'].append('yellow')
                            
                        if hasattr(self.shapes_layer, 'edge_width') and i < len(self.shapes_layer.edge_width):
                            fixed_properties['edge_width'].append(self.shapes_layer.edge_width[i])
                        else:
                            fixed_properties['edge_width'].append(3)
                            
                        if hasattr(self.shapes_layer, 'face_color') and i < len(self.shapes_layer.face_color):
                            fixed_properties['face_color'].append(self.shapes_layer.face_color[i])
                        else:
                            fixed_properties['face_color'].append([0, 0, 0, 0])
                    except:
                        fixed_properties['edge_color'].append('yellow')
                        fixed_properties['edge_width'].append(3)
                        fixed_properties['face_color'].append([0, 0, 0, 0])
                
                # Clear and recreate the layer with fixed shapes
                if fixed_shapes:
                    self.shapes_layer.data = fixed_shapes
                    self.shapes_layer.edge_color = fixed_properties['edge_color']
                    self.shapes_layer.edge_width = fixed_properties['edge_width']
                    self.shapes_layer.face_color = fixed_properties['face_color']
                    
                    print(f"✅ Fixed {len(fixed_shapes)} shapes to be consistently 3D")
                else:
                    print("⚠️ No valid shapes found after conversion")
                
                # Reconnect events
                self.shapes_layer.events.data.connect(self._on_shapes_changed)
            else:
                print("✅ All shapes are already 3D")
                
        except Exception as e:
            # synchronize mappings
            self.synchronize_all_mappings()
            
            print(f"❌ Error ensuring 3D shapes: {e}")
            import traceback
            traceback.print_exc()
            
            # Make sure events are reconnected
            try:
                self.shapes_layer.events.data.connect(self._on_shapes_changed)
            except:
                pass
    
    def get_current_image_points(self, image_index: int, object_id: Optional[int] = None) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Get positive and negative points for a specific image and object"""
        positive_points = []
        negative_points = []
        
        if object_id is not None:
            if object_id not in self.point_annotations_by_object:
                return positive_points, negative_points
            annotations = self.point_annotations_by_object[object_id]
        else:
            object_id = self.current_sam2_object_id
            annotations = self.point_annotations_by_object.get(object_id, [])
        
        for annotation in annotations:
            if annotation.image_index == image_index:
                if annotation.point_type == 'positive':
                    positive_points.append(annotation.roi_local_coords)
                elif annotation.point_type == 'negative':
                    negative_points.append(annotation.roi_local_coords)
        
        return positive_points, negative_points
    
    # ============================================================================
    # SAM2 CORE FUNCTIONS
    # ============================================================================
    
    def extract_roi_as_jpeg(self, image: np.ndarray, roi_params: Dict[str, int]) -> np.ndarray:
        """Extract ROI from image as JPEG-ready array"""
        y1 = roi_params['top_left_y']
        x1 = roi_params['top_left_x']
        y2 = y1 + roi_params['height']
        x2 = x1 + roi_params['width']
        
        y1_safe = max(0, y1)
        x1_safe = max(0, x1)
        y2_safe = min(image.shape[0], y2)
        x2_safe = min(image.shape[1], x2)
        
        roi_cutout = image[y1_safe:y2_safe, x1_safe:x2_safe].copy()
        return roi_cutout.astype(np.uint8)
    
    def crop_mask_to_global_batch(self, crop_mask: np.ndarray, original_shape: Tuple[int, int], 
                                 crop_params: Dict[str, int]) -> np.ndarray:
        """Map segmentation results back to original frame dimensions"""
        y_offset = crop_params['top_left_y']
        x_offset = crop_params['top_left_x']
        
        full_mask = np.zeros(original_shape[:2], dtype=bool)
        
        y_end = min(y_offset + crop_mask.shape[0], full_mask.shape[0])
        x_end = min(x_offset + crop_mask.shape[1], full_mask.shape[1])
        
        full_mask[y_offset:y_end, x_offset:x_end] = crop_mask[:y_end-y_offset, :x_end-x_offset]
        
        return full_mask
    
    def extract_convex_hull_from_mask_batch(self, mask: np.ndarray, frame_idx: int, 
                                           crop_params: Dict[str, int], 
                                           original_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract detailed boundary points from segmentation mask"""
        if mask is None or not mask.any():
            return None
        
        global_mask = self.crop_mask_to_global_batch(mask, original_shape, crop_params)
        
        mask_uint8 = (global_mask * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        contour_points = largest_contour.reshape(-1, 2)
        
        num_points = len(contour_points)
        if num_points > 100:
            step = max(1, num_points // 60)
            sampled_points = contour_points[::step]
        elif num_points > 50:
            step = max(1, num_points // 40)
            sampled_points = contour_points[::step]
        else:
            sampled_points = contour_points
        
        if len(sampled_points) < 3:
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
            sampled_points = approx_polygon.reshape(-1, 2)
            
            if len(sampled_points) < 3:
                return None
        
        sampled_points_yx = np.column_stack((sampled_points[:, 1], sampled_points[:, 0]))
        sampled_points_yx = sampled_points_yx.astype(np.float64)
        
        if sampled_points_yx.shape[1] != 2:
            return None
        
        if not np.all(np.isfinite(sampled_points_yx)):
            return None
        
        return sampled_points_yx

    def _process_single_mask_worker_threaded(self, mask_data: Tuple) -> Tuple[int, int, Optional[np.ndarray]]:
        """
        Threaded worker function for processing a single mask to ROI.

        This function is designed to be called from a ThreadPoolExecutor for parallel
        mask processing. OpenCV operations release the GIL, allowing true parallel execution.

        Args:
            mask_data: Tuple containing (mask, frame_idx, roi_params, original_shape, detail_level, obj_id)

        Returns:
            Tuple of (frame_idx, obj_id, roi_vertices or None)
        """
        mask, frame_idx, roi_params, original_shape, detail_level, obj_id = mask_data

        # Early exit for None masks
        if mask is None:
            return (frame_idx, obj_id, None)

        # Early exit for empty masks (faster than .any() for large arrays)
        if mask.sum() == 0:
            return (frame_idx, obj_id, None)

        try:
            # Map mask back to global coordinates
            global_mask = self.crop_mask_to_global_batch(mask, original_shape, roi_params)

            # Direct conversion to uint8 for OpenCV
            mask_uint8 = (global_mask * 255).astype(np.uint8)

            # Find contours - OpenCV releases GIL here
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return (frame_idx, obj_id, None)

            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Early exit for tiny contours (< 10 pixels area)
            contour_area = cv2.contourArea(largest_contour)
            if contour_area < 10:
                return (frame_idx, obj_id, None)

            # Simplified polygon approximation using approxPolyDP
            perimeter = cv2.arcLength(largest_contour, True)
            epsilon = detail_level * 0.01 * perimeter
            approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Extract points
            polygon_points = approx_polygon.reshape(-1, 2)

            if len(polygon_points) < 3:
                return (frame_idx, obj_id, None)

            # Convert from (x, y) to (y, x) format for napari
            polygon_points_yx = np.column_stack((polygon_points[:, 1], polygon_points[:, 0]))
            polygon_points_yx = polygon_points_yx.astype(np.float64)

            # Validate output
            if polygon_points_yx.shape[1] != 2:
                return (frame_idx, obj_id, None)

            if not np.all(np.isfinite(polygon_points_yx)):
                return (frame_idx, obj_id, None)

            return (frame_idx, obj_id, polygon_points_yx)

        except Exception as e:
            # Log error but don't crash the thread
            print(f"⚠️ Worker error processing mask for frame {frame_idx}, obj {obj_id}: {e}")
            return (frame_idx, obj_id, None)

    def process_masks_to_rois_parallel(self,
                                        masks_per_object: Dict[int, List[np.ndarray]],
                                        images_to_process: List[int],
                                        roi_params: Dict[str, int],
                                        detail_level: float = 0.5,
                                        max_workers: Optional[int] = None) -> Dict[int, Dict[int, List[np.ndarray]]]:
        """
        Process multiple masks to ROIs using parallel execution with ThreadPoolExecutor.

        This method flattens all masks into a single batch for optimal thread distribution,
        then processes them in parallel using a thread pool. OpenCV operations release the GIL,
        enabling true parallel execution.

        Args:
            masks_per_object: Dict mapping obj_id -> list of masks (one per frame)
            images_to_process: List of frame indices being processed
            roi_params: ROI parameters for coordinate mapping
            detail_level: Polygon approximation detail (0.0-1.0, lower = more detail)
            max_workers: Max thread pool workers (None = auto based on CPU count)

        Returns:
            Dict mapping frame_idx -> {obj_id -> [roi_vertices]}
        """
        if self.image_stack is None:
            print("❌ No image stack loaded")
            return {}

        original_shape = self.image_stack.shape[1:3]

        # Flatten all masks into a single batch for optimal thread distribution
        # masks_per_object format: {obj_id: [mask, mask, None, mask, ...]}
        # Each mask index corresponds to images_to_process index
        all_mask_data = []
        for obj_id, masks_list in masks_per_object.items():
            for local_idx, mask in enumerate(masks_list):
                # Skip None masks (frames where SAM2 didn't produce output)
                if mask is not None:
                    # Map local index to actual frame index
                    frame_idx = images_to_process[local_idx]
                    mask_data = (mask, frame_idx, roi_params, original_shape, detail_level, obj_id)
                    all_mask_data.append(mask_data)

        if not all_mask_data:
            print("⚠️ No valid masks to process")
            return {}

        # Determine optimal worker count
        if max_workers is None:
            # Use CPU count but cap at 8 to avoid diminishing returns
            import os
            max_workers = min(os.cpu_count() or 4, 8)

        print(f"🚀 Processing {len(all_mask_data)} masks in parallel (max_workers={max_workers})...")

        # Process masks in parallel
        results = {}
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._process_single_mask_worker_threaded, mask_data): idx
                for idx, mask_data in enumerate(all_mask_data)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    frame_idx, obj_id, roi_vertices = future.result()

                    if roi_vertices is not None:
                        # Initialize nested dict structure if needed
                        if frame_idx not in results:
                            results[frame_idx] = {}
                        if obj_id not in results[frame_idx]:
                            results[frame_idx][obj_id] = []

                        results[frame_idx][obj_id].append(roi_vertices)
                        successful += 1
                    else:
                        failed += 1

                except Exception as e:
                    print(f"⚠️ Future error: {e}")
                    failed += 1

        print(f"✅ Parallel processing complete: {successful} successful, {failed} skipped/failed")

        return results

    def debug_sam2_state(self) -> None:
        """Print detailed SAM2 annotation state for debugging"""
        print("\n" + "="*60)
        print("SAM2 ANNOTATION STATE DEBUG")
        print("="*60)
        
        print(f"SAM2 Mode: {self.sam2_mode}")
        print(f"Session ID: {self.sam2_session_id}")
        print(f"Current Object ID: {self.current_sam2_object_id}")
        
        if self.current_roi_params:
            print(f"Working ROI: {self.current_roi_params['width']}x{self.current_roi_params['height']} at ({self.current_roi_params['top_left_y']}, {self.current_roi_params['top_left_x']})")
        else:
            print("Working ROI: None")
        
        print(f"\nActive Objects: {sorted(self.active_object_ids) if self.active_object_ids else 'None'}")
        
        print(f"\nBox Prompts:")
        if self.sam2_box_prompts_by_object:
            for obj_id, box in self.sam2_box_prompts_by_object.items():
                shape_idx = self.object_to_box_shape_index.get(obj_id, 'Unknown')
                print(f"  Object {obj_id}: {box} (shape index: {shape_idx})")
        else:
            print("  None")
        
        print(f"\nPoint Annotations:")
        if self.point_annotations_by_object:
            for obj_id, annotations in self.point_annotations_by_object.items():
                pos_count = sum(1 for a in annotations if a.point_type == 'positive')
                neg_count = sum(1 for a in annotations if a.point_type == 'negative')
                print(f"  Object {obj_id}: {pos_count} positive, {neg_count} negative points")
        else:
            print("  None")
        
        print("="*60)


    def sam2_propagate_batch(self, image_cutouts: List[np.ndarray], 
                        images_to_process: List[int],
                        points_per_frame_per_object: Dict[int, Dict[int, Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]],
                        box_prompts_per_object: Dict[int, Optional[Tuple[float, float, float, float]]]) -> Dict[int, List[List[np.ndarray]]]:
        """Run SAM2 batch propagation with multiple objects - FIXED for mixed prompt types"""
        print(f"SAM2 batch processing: {len(image_cutouts)} images")
        print(f"Active objects: {sorted(points_per_frame_per_object.keys())}")
        
        try:
            from sam2.build_sam import build_sam2_video_predictor
            import tempfile
            import os
        except ImportError as e:
            print(f"ERROR: SAM2 not available: {e}")
            return {}
        
        temp_dir = tempfile.mkdtemp(prefix="sam2_napari_")
        
        try:
            # Save cropped frames
            for idx, cutout in enumerate(image_cutouts):
                frame_filename = f"{idx:04d}.jpg"
                frame_path = os.path.join(temp_dir, frame_filename)
                cv2.imwrite(frame_path, cv2.cvtColor(cutout, cv2.COLOR_RGB2BGR))
            
            # Initialize SAM2 model
            if not hasattr(self, '_sam2_predictor'):
                if not os.path.exists(SAM2_CHECKPOINT):
                    raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}")
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Initializing SAM2 on device: {device}")
                
                if device.type == 'cuda':
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                self._sam2_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device)
                print("SAM2 model loaded successfully")
            
            predictor = self._sam2_predictor
            
            # Initialize SAM2 state
            state = predictor.init_state(temp_dir)
            predictor.reset_state(state)
            
            # ENHANCED: Better frame assignment and dtype handling for mixed prompt types
            print(f"\n🔧 Processing mixed prompt types with enhanced dtype consistency...")
            
            # Process each object with improved error handling
            for obj_id in sorted(self.active_object_ids):
                print(f"\n🎯 Processing Object {obj_id}...")
                
                points_per_frame = points_per_frame_per_object.get(obj_id, {})
                box_prompt = box_prompts_per_object.get(obj_id)
                
                # FIXED: Improved frame assignment logic for box prompts
                box_prompt_frame = None
                if box_prompt:
                    if points_per_frame:
                        # Use the first frame that has points for this object
                        box_prompt_frame = min(points_per_frame.keys())
                        print(f"  📦 Box prompt will be applied to frame {box_prompt_frame} (has points)")
                    else:
                        # Use first available frame for box-only objects
                        box_prompt_frame = images_to_process[0] if images_to_process else 0
                        print(f"  📦 Box prompt will be applied to frame {box_prompt_frame} (box only)")
                
                # Add prompts with enhanced dtype consistency
                processed_frames = set()
                
                # Process frames with points
                for frame_idx, (positive_points, negative_points) in points_per_frame.items():
                    if frame_idx not in images_to_process:
                        continue
                    
                    local_frame_idx = images_to_process.index(frame_idx)
                    print(f"  Adding prompts to frame {local_frame_idx} (image {frame_idx})")
                    
                    # FIXED: Ensure consistent dtype for point coordinates
                    positive_points_xy = [(float(x), float(y)) for (y, x) in positive_points]
                    negative_points_xy = [(float(x), float(y)) for (y, x) in negative_points]
                    
                    all_points_xy = positive_points_xy + negative_points_xy
                    all_labels = [1] * len(positive_points_xy) + [0] * len(negative_points_xy)
                    
                    try:
                        if box_prompt is not None and frame_idx == box_prompt_frame:
                            # FIXED: Ensure consistent dtypes for mixed prompts
                            if len(all_points_xy) > 0:
                                print(f"    🔗 Box + {len(positive_points_xy)} pos, {len(negative_points_xy)} neg points")
                                
                                # Ensure all arrays are float32
                                points_array = np.array(all_points_xy, dtype=np.float32)
                                labels_array = np.array(all_labels, dtype=np.int32)
                                box_array = np.array(box_prompt, dtype=np.float32)
                                
                                # Validate arrays before passing to SAM2
                                if not self._validate_sam2_inputs(points_array, labels_array, box_array):
                                    print(f"    ❌ Invalid inputs for mixed prompts, skipping")
                                    continue
                                
                                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                    inference_state=state,
                                    frame_idx=local_frame_idx,
                                    obj_id=obj_id,
                                    points=points_array,
                                    labels=labels_array,
                                    box=box_array,
                                )
                            else:
                                print(f"    📦 Box prompt only")
                                
                                # Ensure box array is float32
                                box_array = np.array(box_prompt, dtype=np.float32)
                                
                                if not self._validate_sam2_inputs(box=box_array):
                                    print(f"    ❌ Invalid box input, skipping")
                                    continue
                                
                                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                    inference_state=state,
                                    frame_idx=local_frame_idx,
                                    obj_id=obj_id,
                                    box=box_array,
                                )
                        elif len(all_points_xy) > 0:
                            print(f"    📍 {len(positive_points_xy)} pos, {len(negative_points_xy)} neg points")
                            
                            # Ensure all arrays are consistent dtype
                            points_array = np.array(all_points_xy, dtype=np.float32)
                            labels_array = np.array(all_labels, dtype=np.int32)
                            
                            if not self._validate_sam2_inputs(points_array, labels_array):
                                print(f"    ❌ Invalid point inputs, skipping")
                                continue
                            
                            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=local_frame_idx,
                                obj_id=obj_id,
                                points=points_array,
                                labels=labels_array,
                            )
                        
                        processed_frames.add(frame_idx)
                        
                    except Exception as e:
                        print(f"    ❌ Error adding prompts for Object {obj_id} on frame {frame_idx}: {e}")
                        continue
                
                # FIXED: Handle box-only objects separately to avoid frame conflicts
                if box_prompt and not points_per_frame and box_prompt_frame not in processed_frames:
                    try:
                        local_frame_idx = images_to_process.index(box_prompt_frame) if box_prompt_frame in images_to_process else 0
                        print(f"  📦 Adding box-only prompt to frame {local_frame_idx}")
                        
                        box_array = np.array(box_prompt, dtype=np.float32)
                        
                        if self._validate_sam2_inputs(box=box_array):
                            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=local_frame_idx,
                                obj_id=obj_id,
                                box=box_array,
                            )
                            processed_frames.add(box_prompt_frame)
                        else:
                            print(f"    ❌ Invalid box-only input for Object {obj_id}")
                            
                    except Exception as e:
                        print(f"    ❌ Error adding box-only prompt for Object {obj_id}: {e}")
                        continue
            
            # Propagate across all frames with enhanced error handling
            results_per_object = {obj_id: [[] for _ in range(len(image_cutouts))] for obj_id in self.active_object_ids}
            
            try:
                print(f"\n🚀 Starting SAM2 propagation across {len(image_cutouts)} frames...")
                
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                    for i, obj_id in enumerate(out_obj_ids):
                        if obj_id in results_per_object:
                            crop_mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0)
                            results_per_object[obj_id][out_frame_idx] = [crop_mask]
                
                print(f"\n✅ SAM2 batch processing complete for {len(self.active_object_ids)} objects")
                return results_per_object
                
            except Exception as e:
                print(f"❌ Error during SAM2 propagation: {e}")
                print(f"🔧 This may be due to mixed prompt types causing dtype inconsistency")
                
                # ENHANCED: Provide helpful error message for mixed prompt scenarios
                box_objects = [obj_id for obj_id, box in box_prompts_per_object.items() if box is not None]
                point_objects = [obj_id for obj_id in points_per_frame_per_object.keys()]
                
                if len(box_objects) > 0 and len(point_objects) > 0:
                    mixed_objects = set(box_objects) & set(point_objects)
                    box_only_objects = set(box_objects) - set(point_objects)
                    point_only_objects = set(point_objects) - set(box_objects)
                    
                    print(f"🔍 Prompt analysis:")
                    print(f"  - Box-only objects: {sorted(box_only_objects)}")
                    print(f"  - Point-only objects: {sorted(point_only_objects)}")
                    print(f"  - Mixed prompt objects: {sorted(mixed_objects)}")
                    
                    print(f"💡 Suggestion: Try processing objects with different prompt types separately")
                
                return {}
                
        finally:
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    

    def _validate_sam2_inputs(self, points: Optional[np.ndarray] = None, 
                            labels: Optional[np.ndarray] = None,
                            box: Optional[np.ndarray] = None) -> bool:
        """Validate inputs before passing to SAM2 to prevent dtype errors"""
        
        if points is not None:
            if not isinstance(points, np.ndarray):
                print(f"❌ Points must be numpy array, got {type(points)}")
                return False
            
            if points.dtype != np.float32:
                print(f"❌ Points must be float32, got {points.dtype}")
                return False
            
            if points.ndim != 2 or points.shape[1] != 2:
                print(f"❌ Points must be (N, 2), got {points.shape}")
                return False
            
            if not np.all(np.isfinite(points)):
                print(f"❌ Points contain invalid values (NaN/Inf)")
                return False
        
        if labels is not None:
            if not isinstance(labels, np.ndarray):
                print(f"❌ Labels must be numpy array, got {type(labels)}")
                return False
            
            if labels.dtype != np.int32:
                print(f"❌ Labels must be int32, got {labels.dtype}")
                return False
            
            if labels.ndim != 1:
                print(f"❌ Labels must be 1D, got {labels.shape}")
                return False
            
            if points is not None and len(labels) != len(points):
                print(f"❌ Points and labels length mismatch: {len(points)} vs {len(labels)}")
                return False
        
        if box is not None:
            if not isinstance(box, np.ndarray):
                print(f"❌ Box must be numpy array, got {type(box)}")
                return False
            
            if box.dtype != np.float32:
                print(f"❌ Box must be float32, got {box.dtype}")
                return False
            
            if box.shape != (4,):
                print(f"❌ Box must be (4,), got {box.shape}")
                return False
            
            if not np.all(np.isfinite(box)):
                print(f"❌ Box contains invalid values (NaN/Inf)")
                return False
            
            # Validate box coordinates (x1, y1, x2, y2)
            if box[0] >= box[2] or box[1] >= box[3]:
                print(f"❌ Invalid box coordinates: ({box[0]}, {box[1]}, {box[2]}, {box[3]})")
                return False
        
        return True


    def convert_masks_to_rois(self, masks: List[np.ndarray], 
                            roi_params: Dict[str, int]) -> List[np.ndarray]:
        """Convert binary masks to ROI polygons"""
        rois = []
        for i, mask in enumerate(masks):
            if mask is None or not mask.any():
                continue
                
            if self.image_stack is not None:
                original_shape = self.image_stack.shape[1:3]
            else:
                original_shape = (
                    roi_params['top_left_y'] + roi_params['height'] + 100,
                    roi_params['top_left_x'] + roi_params['width'] + 100
                )
            
            roi_vertices = self.extract_convex_hull_from_mask_batch(
                mask, i, roi_params, original_shape
            )
            
            if roi_vertices is not None:
                if roi_vertices.shape[1] != 2:
                    continue
                
                if len(roi_vertices) < 3:
                    continue
                
                roi_vertices = roi_vertices.astype(np.float64)
                rois.append(roi_vertices)
        
        return rois
    
    # ============================================================================
    # SAM2 WORKFLOW FUNCTIONS
    # ============================================================================
    
    def sam2_seg_button_clicked(self) -> None:
        """Handle SAM2 Seg button click - find untracked rectangles only"""
        # Validate prerequisites
        if self.shapes_layer is None or len(self.shapes_layer.data) == 0:
            print("ERROR: No ROI selected. Please draw a rectangular ROI first.")
            return
        
        if self.image_stack is None:
            print("ERROR: No images loaded.")
            return
        
        # Synchronize all mappings before starting
        self.synchronize_all_mappings()
        
        print(f"\n{'='*60}")
        print(f"🔍 SAM2 Seg Debug - Current state:")
        print(f"{'='*60}")
        print(f"Total shapes in layer: {len(self.shapes_layer.data)}")
        print(f"Tracked shape IDs: {sorted(self.shape_id_to_type.keys())}")
        print(f"Current SAM2 mode: {self.sam2_mode}")
        print(f"Current SAM2 session: {self.sam2_session_id}")
        
        # Debug print all shapes
        print("\nShape inventory:")
        for napari_idx, shape in enumerate(self.shapes_layer.data):
            shape_id = self.napari_index_to_shape_id.get(napari_idx)
            if shape_id is not None:
                shape_type = self.shape_id_to_type.get(shape_id, 'unknown')
                obj_id = self.shape_id_to_object.get(shape_id)
                session_id = self.shape_id_to_session.get(shape_id)
                print(f"  [{napari_idx}] Shape ID {shape_id}: {len(shape)} vertices, "
                    f"type={shape_type}, obj={obj_id}, session={session_id}")
            else:
                print(f"  [{napari_idx}] UNTRACKED: {len(shape)} vertices")
        
        # Find rectangles that are completely untracked OR have no type assigned
        sam2_roi_idx = None
        sam2_roi = None
        candidate_rectangles = []
        
        print("\nSearching for untracked or untyped rectangles...")
        for napari_idx, roi in enumerate(self.shapes_layer.data):
            # Validate shape data
            if not isinstance(roi, np.ndarray):
                print(f"  [{napari_idx}] Skipping - not numpy array")
                continue
            
            if roi.size == 0:
                print(f"  [{napari_idx}] Skipping - empty array")
                continue
            
            if len(roi) != 4:
                print(f"  [{napari_idx}] Skipping - not rectangle ({len(roi)} vertices)")
                continue
            
            # Check if this shape has a shape ID
            shape_id = self.napari_index_to_shape_id.get(napari_idx)
            
            if shape_id is None:
                # No shape ID at all - definitely untracked
                print(f"  [{napari_idx}] ✅ Found completely untracked rectangle!")
                try:
                    roi_params = self.get_rectangle_roi_params(roi)
                    if roi_params['height'] > 0 and roi_params['width'] > 0:
                        if self.validate_roi_bounds(roi_params):
                            candidate_rectangles.append((napari_idx, roi, roi_params))
                            print(f"    Dimensions: {roi_params['width']}x{roi_params['height']} "
                                f"at ({roi_params['top_left_y']}, {roi_params['top_left_x']})")
                except Exception as e:
                    print(f"    ❌ Could not parse as rectangle: {e}")
            else:
                # Has shape ID - check if it has a type
                shape_type = self.shape_id_to_type.get(shape_id)
                
                if shape_type is None or shape_type == 'unknown':
                    # Has ID but no valid type - this is available for SAM2!
                    print(f"  [{napari_idx}] ✅ Found rectangle with shape ID {shape_id} but no/unknown type!")
                    try:
                        roi_params = self.get_rectangle_roi_params(roi)
                        if roi_params['height'] > 0 and roi_params['width'] > 0:
                            if self.validate_roi_bounds(roi_params):
                                candidate_rectangles.append((napari_idx, roi, roi_params))
                                print(f"    Dimensions: {roi_params['width']}x{roi_params['height']} "
                                    f"at ({roi_params['top_left_y']}, {roi_params['top_left_x']})")
                    except Exception as e:
                        print(f"    ❌ Could not parse as rectangle: {e}")
                else:
                    # Has both shape ID and valid type - not available
                    session_id = self.shape_id_to_session.get(shape_id)
                    print(f"  [{napari_idx}] ❌ Already tracked - Shape ID {shape_id}, "
                        f"type={shape_type}, session={session_id}")
        
        # Select the best candidate (prefer larger rectangles)
        if candidate_rectangles:
            if len(candidate_rectangles) == 1:
                sam2_roi_idx, sam2_roi, roi_params = candidate_rectangles[0]
            else:
                # Multiple candidates - choose the largest
                print(f"\nFound {len(candidate_rectangles)} candidate rectangles, selecting largest...")
                sam2_roi_idx, sam2_roi, roi_params = max(
                    candidate_rectangles,
                    key=lambda x: x[2]['width'] * x[2]['height']
                )
            print(f"\n✅ Selected rectangle at index {sam2_roi_idx}")
        else:
            print("\nERROR: No untracked rectangle found. Please draw a new rectangle for SAM2.")
            return
        
        # Start new SAM2 session
        self.sam2_session_id += 1
        print(f"\n🆕 Starting SAM2 session #{self.sam2_session_id}")
        
        # Get rectangle parameters (already have them from validation)
        print(f"Rectangle parameters: {roi_params['width']}x{roi_params['height']} "
            f"at ({roi_params['top_left_y']}, {roi_params['top_left_x']})")
        
        # Check if we need to convert to square
        final_roi = sam2_roi  # Start with original
        final_roi_params = roi_params
        
        if roi_params['height'] != roi_params['width']:
            print(f"\nConverting rectangle to square...")
            
            # Create square ROI from rectangle parameters
            converted_roi = self.convert_to_square_roi(roi_params)
            
            if converted_roi is None:
                print("ERROR: Could not create square ROI within image bounds")
                return
            
            # Get parameters for the converted square ROI
            final_roi_params = self.get_square_roi_params(converted_roi)
            
            # Ensure the converted ROI has the correct dimensions
            if converted_roi.ndim == 2 and converted_roi.shape[1] == 2:
                # Add Z coordinate if missing
                z_coord = self.current_index
                z_coords = np.full((len(converted_roi), 1), z_coord, dtype=np.float64)
                converted_roi = np.hstack([z_coords, converted_roi])
                print(f"Converted square ROI to 3D with Z={z_coord}")

            try:
                # Disconnect events to prevent recursion
                self.shapes_layer.events.data.disconnect(self._on_shapes_changed)
                
                # Get current shapes and ensure they're all the same dimensionality
                current_shapes = list(self.shapes_layer.data)
                
                # Check dimensionality of existing shapes
                ndim = None
                for i, shape in enumerate(current_shapes):
                    if isinstance(shape, np.ndarray) and shape.size > 0:
                        if ndim is None:
                            ndim = shape.shape[1] if shape.ndim == 2 else 0
                        print(f"Shape {i}: dimensions = {shape.shape}")
                
                # Ensure our converted ROI matches the dimensionality
                if ndim == 3 and converted_roi.shape[1] == 2:
                    # Need to add Z dimension
                    z_coord = self.current_index
                    z_coords = np.full((len(converted_roi), 1), z_coord, dtype=np.float64)
                    converted_roi = np.hstack([z_coords, converted_roi])
                elif ndim == 2 and converted_roi.shape[1] == 3:
                    # Need to remove Z dimension (rare case)
                    converted_roi = converted_roi[:, 1:]
                
                # Method 1: Direct update
                try:
                    current_shapes[sam2_roi_idx] = converted_roi
                    self.shapes_layer.data = current_shapes
                    final_roi = converted_roi
                    print("✅ Successfully updated rectangle to square (method 1)")
                    
                except Exception as e1:
                    print(f"Method 1 failed: {e1}")
                    
                    # Method 2: Remove and add with proper dimensions
                    try:
                        # Store properties of all shapes
                        all_properties = {
                            'edge_color': list(self.shapes_layer.edge_color),
                            'edge_width': list(self.shapes_layer.edge_width),
                            'face_color': list(self.shapes_layer.face_color)
                        }
                        
                        # Remove the rectangle
                        self.shapes_layer.selected_data = {sam2_roi_idx}
                        self.shapes_layer.remove_selected()
                        
                        # Add the square ROI with explicit 3D specification
                        self.shapes_layer.add(
                            converted_roi,
                            shape_type='rectangle',
                            edge_color='lime',
                            edge_width=4,
                            face_color=[0, 0, 0, 0],
                            ndim=3  # Force 3D
                        )
                        
                        # Update index (now at the end)
                        sam2_roi_idx = len(self.shapes_layer.data) - 1
                        final_roi = converted_roi
                        print("✅ Successfully updated rectangle to square (method 2)")
                        
                    except Exception as e2:
                        print(f"Method 2 also failed: {e2}")
                        # Safely reconnect events
                        self._safely_reconnect_shape_events()
                        return

                # Safely reconnect events
                self._safely_reconnect_shape_events()
                
            except Exception as e:
                print(f"ERROR: Failed to update rectangle to square: {e}")
                import traceback
                traceback.print_exc()
                # Ensure events are reconnected
                self._safely_reconnect_shape_events()
                return
        
        # Create new shape ID for the working ROI
        working_roi_shape_id = self._next_shape_id
        self._next_shape_id += 1
        
        print(f"\nAssigning shape ID {working_roi_shape_id} to working ROI at index {sam2_roi_idx}")
        
        # Store all the tracking information
        self.current_square_roi = final_roi
        self.current_roi_params = final_roi_params
        self.sam2_mode = "annotation"
        self.sam2_working_roi_id = sam2_roi_idx
        
        # Check if the selected rectangle already has a shape ID
        existing_shape_id = self.napari_index_to_shape_id.get(sam2_roi_idx)
        
        if existing_shape_id is not None:
            # Reuse existing shape ID but update its type
            working_roi_shape_id = existing_shape_id
            print(f"\nReusing existing shape ID {working_roi_shape_id} for working ROI")
            
            # Clean up any stale type/object/session data
            if existing_shape_id in self.shape_id_to_type:
                old_type = self.shape_id_to_type[existing_shape_id]
                print(f"  Updating type from '{old_type}' to 'working_roi'")
        else:
            # Create new shape ID
            working_roi_shape_id = self._next_shape_id
            self._next_shape_id += 1
            print(f"\nAssigning new shape ID {working_roi_shape_id} to working ROI")
        
        # Update all mappings
        self.shape_id_to_napari_index[working_roi_shape_id] = sam2_roi_idx
        self.napari_index_to_shape_id[sam2_roi_idx] = working_roi_shape_id
        self.shape_id_to_coordinates[working_roi_shape_id] = final_roi.copy()
        self.shape_id_to_object[working_roi_shape_id] = -1  # Special ID for working ROI
        self.shape_id_to_type[working_roi_shape_id] = 'working_roi'
        self.shape_id_to_session[working_roi_shape_id] = self.sam2_session_id
        
        # Clear SAM2 state for new session
        print("\nClearing SAM2 state for new session...")
        self.point_annotations_by_object.clear()
        self.sam2_box_prompts_by_object.clear()
        self.active_object_ids.clear()
        self.current_sam2_object_id = 1
        
        # Clear point tracking
        self.positive_point_to_object.clear()
        self.negative_point_to_object.clear()
        self.processed_positive_points.clear()
        self.processed_negative_points.clear()
        
        # Update ROI appearance
        print("Updating ROI appearance...")
        try:
            if hasattr(self.shapes_layer, 'edge_color'):
                colors = list(self.shapes_layer.edge_color)
                while len(colors) <= sam2_roi_idx:
                    colors.append('yellow')
                colors[sam2_roi_idx] = 'lime'
                self.shapes_layer.edge_color = colors
                
            if hasattr(self.shapes_layer, 'edge_width'):
                widths = list(self.shapes_layer.edge_width)
                while len(widths) <= sam2_roi_idx:
                    widths.append(3)
                widths[sam2_roi_idx] = 4
                self.shapes_layer.edge_width = widths
                
            print("✅ ROI appearance updated")
        except Exception as e:
            # synchronize mappings
            self.synchronize_all_mappings()

            print(f"⚠️ Warning: Could not update ROI appearance: {e}")
        
        # Initialize point layers
        print("\nInitializing point layers...")
        self.initialize_point_layers()
        self.clear_point_layers()
        
        # Save the working ROI as an annotation
        img_idx = int(final_roi[0, 0]) if final_roi.shape[1] == 3 else self.current_index
        self.save_annotation_with_shape_id(
            img_idx,
            'working_roi',
            final_roi,
            shape_id=working_roi_shape_id,
            object_id=-1,
            metadata={'session_id': self.sam2_session_id}
        )
        
        # Final status
        print(f"\n{'='*60}")
        print(f"✅ SAM2 annotation mode enabled!")
        print(f"📐 Square ROI: top_left=({final_roi_params['top_left_y']}, {final_roi_params['top_left_x']}), "
            f"size=({final_roi_params['height']}x{final_roi_params['width']})")
        print(f"🎯 Current Object ID: {self.current_sam2_object_id}")
        print(f"📝 Working ROI tracked with shape ID {working_roi_shape_id} at napari index {sam2_roi_idx}")
        print(f"🆔 Session ID: {self.sam2_session_id}")
        print(f"{'='*60}")
        
        print("\n🎯 SAM2 ANNOTATION INSTRUCTIONS:")
        print("1. Add positive points (green) inside objects")
        print("2. Add negative points (red) outside objects")
        print("3. Draw box prompts inside the green square")
        print("4. Change 'SAM2 Object ID' to annotate multiple objects")
        print("5. Click 'Propagate' to process all annotated objects")
    

    def check_event_handlers(self) -> None:
        """Diagnostic: Check if event handlers are connected"""
        if self.shapes_layer is None:
            print("❌ shapes_layer is None")
            return

        print("\n🔍 EVENT HANDLER DIAGNOSTIC:")
        print(f"   shapes_layer exists: ✅")

        # Check data event
        try:
            callbacks = self.shapes_layer.events.data.callbacks
            print(f"   data event callbacks: {len(callbacks)} registered")
            for i, cb in enumerate(callbacks):
                print(f"      [{i}] {cb}")
                if hasattr(cb, '__self__') and cb.__self__ == self:
                    if cb.__name__ == '_on_shapes_changed':
                        print(f"         ✅ Found _on_shapes_changed!")
        except Exception as e:
            print(f"   ❌ Error checking data callbacks: {e}")

        # Check selected_data event
        try:
            callbacks = self.shapes_layer.events.selected_data.callbacks
            print(f"   selected_data event callbacks: {len(callbacks)} registered")
            for i, cb in enumerate(callbacks):
                print(f"      [{i}] {cb}")
                if hasattr(cb, '__self__') and cb.__self__ == self:
                    if cb.__name__ == '_on_selection_changed':
                        print(f"         ✅ Found _on_selection_changed!")
        except Exception as e:
            print(f"   ❌ Error checking selected_data callbacks: {e}")

    def _safely_reconnect_shape_events(self) -> None:
        """Safely reconnect shape change events (only if not already connected)"""
        if self.shapes_layer is None:
            return

        print("🔧 Attempting to safely reconnect shape events...")

        try:
            # Try to disconnect first - if it fails, it's not connected
            self.shapes_layer.events.data.disconnect(self._on_shapes_changed)
            print("   Disconnected existing handler")
            # If disconnect succeeded, reconnect
            self.shapes_layer.events.data.connect(self._on_shapes_changed)
            print("   ✅ Reconnected handler")
        except:
            # Not connected, so connect it
            try:
                self.shapes_layer.events.data.connect(self._on_shapes_changed)
                print("   ✅ Connected handler (was not connected)")
            except Exception as e:
                # Already connected, nothing to do
                print(f"   ℹ️ Handler already connected or error: {e}")

    def _on_selection_changed(self) -> None:
        """Capture shape state when selection changes"""
        if self.shapes_layer is None or self._modifying_shapes:
            return

        try:
            selected = self.shapes_layer.selected_data

            # CRITICAL FIX: Only take snapshots of shapes that are in our tracking system
            # This prevents new shapes (that napari auto-selects) from being treated as modifications
            tracked_selected = {idx for idx in selected if idx in self.napari_index_to_shape_id}

            # If no tracked shapes are selected, don't create a snapshot
            if not tracked_selected:
                print(f"📸 Selection changed but no tracked shapes selected - skipping snapshot")
                return

            # Take snapshot of selected shapes
            self.shape_selection_snapshot = {
                'selected_indices': tracked_selected,
                'shapes_data': {},
                'snapshot_time': time.time()
            }

            # Store detailed data for each selected shape
            for idx in tracked_selected:
                if idx < len(self.shapes_layer.data):
                    shape_data = self.shapes_layer.data[idx]
                    shape_id = self.napari_index_to_shape_id.get(idx)

                    if shape_id is not None:
                        # Get all metadata for this shape
                        metadata = {
                            'shape_id': shape_id,
                            'type': self.shape_id_to_type.get(shape_id),
                            'object_id': self.shape_id_to_object.get(shape_id),
                            'session_id': self.shape_id_to_session.get(shape_id),
                            'label': None,
                            'label_name': None
                        }
                        
                        # Extract label from stored annotation data
                        shape_type = metadata['type']
                        if shape_type in ['user_roi', 'sam2_result']:
                            # Find the annotation record for this shape
                            for record in self.annotation_history:
                                if (record.metadata and 
                                    record.metadata.get('shape_id') == shape_id):
                                    metadata['label'] = record.metadata.get('label')
                                    metadata['label_name'] = record.metadata.get('label_name')
                                    break
                        
                        self.shape_selection_snapshot['shapes_data'][idx] = {
                            'shape_id': shape_id,
                            'coordinates': shape_data.copy(),
                            'metadata': metadata
                        }
                        
                        print(f"📸 Snapshot taken for shape {idx} (ID: {shape_id}, type: {metadata['type']})")

            if len(tracked_selected) > 0:
                print(f"📸 Selection snapshot: {len(tracked_selected)} tracked shapes selected")
                
        except Exception as e:
            print(f"❌ Error in selection snapshot: {e}")
            import traceback
            traceback.print_exc()


    def _detect_shape_modification(self, old_shapes: List, new_shapes: List) -> Dict[int, int]:
        """Detect which shapes were modified based on selection snapshot

        Returns:
            Dict mapping old_index -> new_index for modified shapes
        """
        modifications = {}

        if not self.shape_selection_snapshot['selected_indices']:
            return modifications

        # CRITICAL FIX: If shape count changed, these are NEW shapes, not modifications
        # This prevents newly drawn shapes from being incorrectly treated as modifications
        if len(new_shapes) != len(old_shapes):
            print(f"   Shape count changed ({len(old_shapes)} -> {len(new_shapes)}), treating as new shapes")
            return modifications

        # Check if this is a recent selection (within 30 seconds)
        if (self.shape_selection_snapshot['snapshot_time'] and
            time.time() - self.shape_selection_snapshot['snapshot_time'] > 30):
            print("⚠️ Selection snapshot too old, treating as new shapes")
            return modifications

        selected_indices = self.shape_selection_snapshot['selected_indices']
        
        # Simple case: single shape selected
        if len(selected_indices) == 1:
            old_idx = next(iter(selected_indices))
            
            # Check if the shape at this index changed
            if old_idx < len(old_shapes) and old_idx < len(new_shapes):
                old_shape = old_shapes[old_idx]
                new_shape = new_shapes[old_idx]
                
                # If coordinates changed but shape still exists at same index
                if not self._shapes_match(old_shape, new_shape):
                    modifications[old_idx] = old_idx
                    print(f"✏️ Detected modification of shape at index {old_idx}")
        
        # Multiple shapes selected - only handle if indices remain stable
        else:
            for old_idx in selected_indices:
                if old_idx < len(old_shapes) and old_idx < len(new_shapes):
                    old_shape = old_shapes[old_idx]
                    new_shape = new_shapes[old_idx]
                    
                    if not self._shapes_match(old_shape, new_shape):
                        modifications[old_idx] = old_idx
                        print(f"✏️ Detected modification of shape at index {old_idx}")
        
        return modifications


    def _shapes_match(self, shape1: np.ndarray, shape2: np.ndarray) -> bool:
        """Check if two shapes are the same by comparing coordinates"""
        if shape1.shape != shape2.shape:
            return False
        
        # Use a small tolerance for floating point comparison
        return np.allclose(shape1, shape2, rtol=1e-5, atol=1e-5)


    def synchronize_all_mappings(self) -> None:
        """Complete synchronization of all tracking systems"""
        if self.shapes_layer is None:
            return
            
        print("🔧 Synchronizing all mappings...")
        
        current_shapes = self.shapes_layer.data
        
        # Step 1: Clean up shape IDs that no longer exist
        shape_ids_to_remove = []
        for shape_id, stored_coords in list(self.shape_id_to_coordinates.items()):
            found = False
            for shape in current_shapes:
                if self._shapes_match(stored_coords, shape):
                    found = True
                    break
            if not found:
                shape_ids_to_remove.append(shape_id)
        
        # Remove orphaned shape IDs
        for shape_id in shape_ids_to_remove:
            print(f"  Removing orphaned shape ID {shape_id}")
            # Remove from ALL mappings
            for mapping in [self.shape_id_to_napari_index, self.shape_id_to_type, 
                        self.shape_id_to_object, self.shape_id_to_session, 
                        self.shape_id_to_coordinates]:
                if shape_id in mapping:
                    del mapping[shape_id]
        
        # Step 2: Rebuild napari index mapping
        self.napari_index_to_shape_id.clear()
        self.shape_id_to_napari_index.clear()
        
        for napari_idx, current_shape in enumerate(current_shapes):
            # Find matching shape ID
            for shape_id, stored_coords in self.shape_id_to_coordinates.items():
                if self._shapes_match(stored_coords, current_shape):
                    self.napari_index_to_shape_id[napari_idx] = shape_id
                    self.shape_id_to_napari_index[shape_id] = napari_idx
                    break
        
        # Step 3: Clean up legacy mappings
        self.shape_to_object_mapping.clear()
        self.shape_to_type_mapping.clear()
        self.shape_to_session_mapping.clear()
        
        # Step 4: Update working ROI ID
        if self.sam2_working_roi_id is not None:
            found = False
            for shape_id, shape_type in self.shape_id_to_type.items():
                if shape_type == 'working_roi':
                    napari_idx = self.shape_id_to_napari_index.get(shape_id)
                    if napari_idx is not None and napari_idx < len(current_shapes):
                        self.sam2_working_roi_id = napari_idx
                        found = True
                        break
            if not found:
                self.sam2_working_roi_id = None
                print("  Working ROI no longer exists")
        
        print("✅ Synchronization complete")
        # Report ALL shape IDs that have coordinates, not just those with types
        all_shape_ids = sorted(self.shape_id_to_coordinates.keys())
        typed_shape_ids = sorted(self.shape_id_to_type.keys())
        print(f"  All shape IDs with coordinates: {all_shape_ids}")
        print(f"  Shape IDs with assigned types: {typed_shape_ids}")
        print(f"  Current napari indices: {sorted(self.napari_index_to_shape_id.keys())}")


    def validate_shape_layer_integrity(self) -> bool:
        """Validate that the shapes layer and mappings are consistent"""
        if self.shapes_layer is None:
            return True
            
        errors = []
        current_shape_count = len(self.shapes_layer.data)
        
        # Check for valid shape indices in mappings
        for mapping_name, mapping in [
            ("shape_to_object_mapping", self.shape_to_object_mapping),
            ("shape_to_type_mapping", self.shape_to_type_mapping),
            ("shape_to_session_mapping", self.shape_to_session_mapping)
        ]:
            for shape_idx in mapping.keys():
                if shape_idx >= current_shape_count:
                    errors.append(f"{mapping_name}: index {shape_idx} >= {current_shape_count}")
        
        # Check for valid shapes
        for idx, shape in enumerate(self.shapes_layer.data):
            if not isinstance(shape, np.ndarray):
                errors.append(f"Shape {idx}: not numpy array ({type(shape)})")
            elif shape.size == 0:
                errors.append(f"Shape {idx}: empty array")
            elif len(shape) < 3:
                errors.append(f"Shape {idx}: too few vertices ({len(shape)})")
            elif not np.all(np.isfinite(shape)):
                errors.append(f"Shape {idx}: contains invalid coordinates")
        
        if errors:
            print("❌ Shape layer integrity check failed:")
            for error in errors[:10]:  # Limit to first 10 errors
                print(f"   - {error}")
            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more errors")
            return False
        else:
            print("✅ Shape layer integrity check passed")
            return True


    def emergency_cleanup_shapes(self) -> None:
        """Emergency cleanup of corrupted shapes and mappings"""
        if self.shapes_layer is None:
            return
            
        print("🚨 Performing emergency cleanup of shapes layer...")
        
        try:
            # Temporarily disconnect events
            self.shapes_layer.events.data.disconnect(self._on_shapes_changed)
            
            # Validate and clean each shape
            valid_shapes = []
            valid_mappings = {}
            
            for old_idx, shape in enumerate(self.shapes_layer.data):
                try:
                    # Validate shape
                    if (isinstance(shape, np.ndarray) and 
                        shape.size > 0 and 
                        len(shape) >= 3 and
                        len(shape) < 1000 and
                        np.all(np.isfinite(shape))):
                        
                        valid_shapes.append(shape)
                        new_idx = len(valid_shapes) - 1
                        
                        # Preserve mappings for valid shapes
                        if old_idx in self.shape_to_object_mapping:
                            valid_mappings[new_idx] = {
                                'object': self.shape_to_object_mapping[old_idx],
                                'type': self.shape_to_type_mapping.get(old_idx, 'user_roi'),
                                'session': self.shape_to_session_mapping.get(old_idx, 0)
                            }
                        
                        print(f"✅ Kept shape {old_idx} -> {new_idx}")
                    else:
                        print(f"❌ Removed invalid shape {old_idx}")
                        
                except Exception as e:
                    # synchronize mappings
                    self.synchronize_all_mappings()

                    print(f"❌ Error processing shape {old_idx}: {e}")
            
            # Update the layer with valid shapes
            self.shapes_layer.data = valid_shapes
            
            # Rebuild mappings
            self.shape_to_object_mapping.clear()
            self.shape_to_type_mapping.clear()
            self.shape_to_session_mapping.clear()
            
            for new_idx, mapping in valid_mappings.items():
                self.shape_to_object_mapping[new_idx] = mapping['object']
                self.shape_to_type_mapping[new_idx] = mapping['type']
                self.shape_to_session_mapping[new_idx] = mapping['session']
            
            # Update working ROI ID if it exists
            if self.sam2_working_roi_id is not None:
                found_working_roi = False
                for idx, shape_type in self.shape_to_type_mapping.items():
                    if shape_type == 'working_roi':
                        self.sam2_working_roi_id = idx
                        found_working_roi = True
                        break
                if not found_working_roi:
                    self.sam2_working_roi_id = None
            
            # Reconnect events
            self.shapes_layer.events.data.connect(self._on_shapes_changed)
            
            print(f"✅ Emergency cleanup complete: {len(valid_shapes)} valid shapes retained")
            
        except Exception as e:
            # synchronize mappings
            self.synchronize_all_mappings()

            print(f"❌ Emergency cleanup failed: {e}")
            # Make sure events are reconnected
            try:
                self.shapes_layer.events.data.connect(self._on_shapes_changed)
            except:
                pass


    def propagate_button_clicked(self, num_images: int, direction: str = "forward") -> None:
        """Handle Propagate button click with enhanced error handling"""
        if self.sam2_mode not in ["annotation", "refining"]:
            print("ERROR: Must click 'SAM2 Seg' first")
            return
        
        # synchronize mappings
        self.synchronize_all_mappings()
            
        if self.current_roi_params is None:
            print("ERROR: No active square ROI.")
            return
        
        if self.image_stack is None:
            print("ERROR: No images loaded.")
            return
        
        self.validate_annotation_state()
        
        # Ensure current object is active if it has annotations
        obj_has_points = (self.current_sam2_object_id in self.point_annotations_by_object and 
                        len(self.point_annotations_by_object[self.current_sam2_object_id]) > 0)
        obj_has_box = (self.current_sam2_object_id in self.sam2_box_prompts_by_object and 
                    self.sam2_box_prompts_by_object[self.current_sam2_object_id] is not None)
        
        if obj_has_points or obj_has_box:
            self.active_object_ids.add(self.current_sam2_object_id)
        
        if len(self.active_object_ids) == 0:
            print("ERROR: No objects with annotations found.")
            return
        
        # ENHANCED: Check for mixed prompt types and warn user
        box_objects = set()
        point_objects = set()
        
        for obj_id in self.active_object_ids:
            if obj_id in self.sam2_box_prompts_by_object and self.sam2_box_prompts_by_object[obj_id] is not None:
                box_objects.add(obj_id)
            if obj_id in self.point_annotations_by_object and len(self.point_annotations_by_object[obj_id]) > 0:
                point_objects.add(obj_id)
        
        mixed_objects = box_objects & point_objects
        box_only_objects = box_objects - point_objects  
        point_only_objects = point_objects - box_objects
        
        if len(box_only_objects) > 0 and len(point_only_objects) > 0:
            print(f"\n⚠️ WARNING: Mixed prompt types detected!")
            print(f"   Box-only objects: {sorted(box_only_objects)}")
            print(f"   Point-only objects: {sorted(point_only_objects)}")
            if mixed_objects:
                print(f"   Mixed prompt objects: {sorted(mixed_objects)}")
            print(f"   This may cause dtype errors in SAM2. Consider processing separately if errors occur.\n")
        
        # Determine images to process
        if self.sam2_mode == "refining":
            images_with_points = set()
            for obj_id in self.active_object_ids:
                if obj_id in self.point_annotations_by_object:
                    for ann in self.point_annotations_by_object[obj_id]:
                        images_with_points.add(ann.image_index)
            
            images_to_process = sorted(list(images_with_points.union(self.propagated_images)))
            images_to_process = [i for i in images_to_process if i < len(self.image_stack)]
            print(f"Refining segmentation on {len(images_to_process)} images")
        else:
            if direction == "forward":
                start_idx = self.current_index
                end_idx = min(start_idx + num_images, len(self.image_stack))
                images_to_process = list(range(start_idx, end_idx))
            else:
                start_idx = max(0, self.current_index - num_images + 1)
                end_idx = self.current_index + 1
                images_to_process = list(range(start_idx, end_idx))
            
            print(f"Propagating {direction} across {len(images_to_process)} images")
        
        print(f"\n📊 Active objects: {sorted(self.active_object_ids)}")
        
        # Collect points per frame per object
        points_per_frame_per_object = {}
        
        for obj_id in self.active_object_ids:
            points_per_frame_per_object[obj_id] = {}
            
            for img_idx in images_to_process:
                img_positive, img_negative = self.get_current_image_points(img_idx, obj_id)
                if img_positive or img_negative:
                    points_per_frame_per_object[obj_id][img_idx] = (img_positive, img_negative)
        
        # For objects with only box prompts
        for obj_id in self.active_object_ids:
            if obj_id not in points_per_frame_per_object or not points_per_frame_per_object[obj_id]:
                if obj_id in self.sam2_box_prompts_by_object and self.sam2_box_prompts_by_object[obj_id]:
                    points_per_frame_per_object[obj_id] = {images_to_process[0]: ([], [])}
        
        # Extract image cutouts
        print("🔍 Extracting ROI cutouts...")
        image_cutouts = []
        for idx in images_to_process:
            image = self.image_stack[idx]
            cutout = self.extract_roi_as_jpeg(image, self.current_roi_params)
            image_cutouts.append(cutout)
        
        # Run SAM2 with enhanced error handling
        try:
            results_per_object = self.sam2_propagate_batch(
                image_cutouts, 
                images_to_process,
                points_per_frame_per_object,
                self.sam2_box_prompts_by_object
            )
            
            if not results_per_object:
                print("❌ SAM2 processing failed - no results returned")
                return

            print(f"✅ SAM2 processing completed")

            # Convert masks to ROIs using parallel processing
            # First, restructure results_per_object for the parallel processor
            # results_per_object format: {obj_id: [[masks_frame_0], [masks_frame_1], ...]}
            # Restructure SAM2 results for parallel processing
            # SAM2 returns: {obj_id: [[mask], [mask], [], ...]} where each element is a list
            # We need: {obj_id: [mask, mask, None, ...]} for the parallel processor
            print(f"\n📦 Restructuring SAM2 results for parallel processing...")
            print(f"  Processing {len(results_per_object)} objects")

            masks_per_object = {}
            for obj_id, batch_results in results_per_object.items():
                masks_per_object[obj_id] = []
                valid_masks_count = 0

                for frame_masks in batch_results:
                    # frame_masks is either [crop_mask] or [] (empty list)
                    if frame_masks:  # Non-empty list
                        masks_per_object[obj_id].append(frame_masks[0])
                        valid_masks_count += 1
                    else:  # Empty list - SAM2 produced no mask for this frame
                        masks_per_object[obj_id].append(None)

                print(f"  Object {obj_id}: {valid_masks_count}/{len(batch_results)} valid masks")

            # Use parallel processing for mask-to-ROI conversion
            all_rois = self.process_masks_to_rois_parallel(
                masks_per_object,
                images_to_process,
                self.current_roi_params,
                detail_level=0.5  # Balanced detail level
            )

            # Update propagated images tracking
            for idx in all_rois.keys():
                self.propagated_images.add(idx)

            # Process results
            self.process_sam2_results_multi_object(all_rois)
            
            if self.sam2_mode == "refining":
                print(f"🔧 Refinement complete!")
            else:
                self.sam2_mode = "propagated"
                print(f"🚀 {direction.capitalize()} propagation complete!")
            
        except Exception as e:
            print(f"❌ ERROR in SAM2 propagation: {e}")
            print(f"🔧 If this is a mixed prompt type error, try:")
            print(f"   1. Process objects with different prompt types separately")
            print(f"   2. Use only one prompt type per session")
            print(f"   3. Check that all prompts are valid")
            import traceback
            traceback.print_exc()
    

    def process_sam2_results_multi_object(self, sam2_rois: Dict[int, Dict[int, List[np.ndarray]]]) -> None:
        """Process ROIs from multiple objects with different colors"""
        if self.shapes_layer is None:
            return
        
        # synchronize mappings
        self.synchronize_all_mappings()

        object_colors = [
            'cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray'
        ]
        
        try:
            total_rois = 0
            rois_added = 0
            
            for img_idx, objects_data in sam2_rois.items():
                for obj_id, roi_list in objects_data.items():
                    color_idx = (obj_id - 1) % len(object_colors)
                    obj_color = object_colors[color_idx]
                    
                    for roi_idx, roi in enumerate(roi_list):
                        total_rois += 1
                        try:
                            # Debug: check ROI properties
                            print(f"🔍 Processing ROI for Object {obj_id}, Image {img_idx}: shape={roi.shape}, type={type(roi)}")
                            
                            # Validate ROI before adding
                            if roi is None:
                                print(f"  ⚠️ ROI is None, skipping")
                                continue
                            
                            if not isinstance(roi, np.ndarray):
                                print(f"  ⚠️ ROI is not numpy array: {type(roi)}")
                                roi = np.array(roi)
                            
                            if roi.size == 0:
                                print(f"  ⚠️ ROI is empty, skipping")
                                continue
                            
                            if len(roi.shape) != 2 or roi.shape[1] != 2:
                                print(f"  ⚠️ ROI has wrong shape: {roi.shape}, skipping")
                                continue
                            
                            if len(roi) < 3:
                                print(f"  ⚠️ ROI has too few points: {len(roi)}, skipping")
                                continue
                            
                            self.safely_add_roi(
                                roi,
                                target_images=[img_idx],
                                object_id=obj_id,
                                edge_color=obj_color,
                                edge_width=2,
                                face_color=[0, 0, 0, 0]
                            )
                            rois_added += 1
                            print(f"  ✅ Successfully added ROI for Object {obj_id} on image {img_idx+1} ({obj_color})")
                        except Exception as e:
                            # synchronize mappings
                            self.synchronize_all_mappings()

                            print(f"  ❌ Error adding ROI for Object {obj_id}: {e}")
                            import traceback
                            traceback.print_exc()
            
            print(f"📊 ROI Addition Summary: {rois_added}/{total_rois} ROIs added successfully")
            if rois_added < total_rois:
                print(f"⚠️ {total_rois - rois_added} ROIs failed to add due to format issues")
            
            self.validate_annotation_state()
            
        except Exception as e:
            print(f"❌ ERROR in process_sam2_results_multi_object: {e}")
            import traceback
            traceback.print_exc()
    
    def refine_seg_button_clicked(self) -> None:
        """Handle Refine Seg button click"""
        if self.sam2_mode not in ["propagated", "refining"]:
            print("ERROR: Must run 'Propagate' first before refining.")
            return
        
        # synchronize mappings
        self.synchronize_all_mappings()

        if self.current_roi_params is None:
            print("ERROR: No active ROI.")
            return
        
        self.sam2_mode = "refining"
        
        self.initialize_point_layers()
        
        if self.positive_points_layer is not None:
            self._last_positive_count = len(self.positive_points_layer.data)
        if self.negative_points_layer is not None:
            self._last_negative_count = len(self.negative_points_layer.data)
        
        print("🔧 Refinement mode enabled!")
        print("Add more points and click 'Propagate' to refine segmentation")
    

    def _convert_sam2_results_to_user_rois(self, session_id: int) -> None:
        """Convert SAM2 result annotations to user ROI annotations with current label"""
        print(f"🔄 Converting SAM2 result annotations to user ROIs with label {self.selected_label}...")
        
        converted_count = 0
        
        # Update annotation history
        for record in self.annotation_history:
            if (record.annotation_type == 'sam2_result' and 
                record.metadata and 
                record.metadata.get('session_id') == session_id):
                
                # Convert to user_roi
                record.annotation_type = 'user_roi'
                record.object_id = 0  # User ROI object ID
                
                # Update metadata with current label
                if record.metadata:
                    record.metadata['label'] = self.selected_label
                    record.metadata['label_name'] = self.label_names[self.selected_label - 1]
                    record.metadata['finalized_from_sam2'] = True
                    # Remove SAM2-specific metadata
                    if 'session_id' in record.metadata:
                        del record.metadata['session_id']
                
                converted_count += 1
        
        # Update per-image annotations
        for img_idx in self.all_annotations.keys():
            for ann in self.all_annotations[img_idx]:
                if (ann.get('type') == 'sam2_result' and
                    ann.get('metadata', {}).get('session_id') == session_id):
                    
                    # Convert to user_roi
                    ann['type'] = 'user_roi'
                    ann['object_id'] = 0
                    ann['label'] = self.selected_label
                    ann['label_name'] = self.label_names[self.selected_label - 1]
                    
                    # Update metadata
                    if 'metadata' in ann:
                        ann['metadata']['label'] = self.selected_label
                        ann['metadata']['label_name'] = self.label_names[self.selected_label - 1]
                        ann['metadata']['finalized_from_sam2'] = True
                        if 'session_id' in ann['metadata']:
                            del ann['metadata']['session_id']
        
        print(f"✅ Converted {converted_count} SAM2 results to labeled user ROIs")


    def finalize_sam2_seg_button_clicked(self) -> None:
        """Handle Finalize SAM2 Seg button click - finalizes SAM2 ROIs with current label and removes working area"""
        if getattr(self, 'sam2_mode', 'idle') == 'idle':
            print("No SAM2 segmentation to finalize.")
            return
        
        current_session = getattr(self, 'sam2_session_id', 0)
        print(f"🎨 Finalizing SAM2 ROIs from session {current_session} with label {self.selected_label} ({self.label_names[self.selected_label - 1]})...")
        
        # STEP 1: Finalize SAM2 result ROIs with current label color
        if self.shapes_layer is not None:
            sam2_result_shapes = []
            working_area_shapes = []
            
            # Find SAM2 results and working area shapes from current session
            for shape_id, session_id in self.shape_id_to_session.items():
                if session_id == current_session:
                    shape_type = self.shape_id_to_type.get(shape_id)
                    napari_idx = self.shape_id_to_napari_index.get(shape_id)
                    
                    if napari_idx is not None and napari_idx < len(self.shapes_layer.data):
                        if shape_type == 'sam2_result':
                            sam2_result_shapes.append((shape_id, napari_idx))
                            print(f"📝 Found SAM2 result (shape ID {shape_id}) at napari index {napari_idx}")
                        elif shape_type in ['working_roi', 'box_prompt']:
                            working_area_shapes.append(napari_idx)
                            print(f"📝 Found {shape_type} (shape ID {shape_id}) at napari index {napari_idx}")
            
            # FINALIZE SAM2 RESULTS: Apply label color and convert to user ROIs
            if sam2_result_shapes:
                print(f"🎨 Finalizing {len(sam2_result_shapes)} SAM2 results with current label...")
                
                for shape_id, napari_idx in sam2_result_shapes:
                    # Apply label color with edge width 10
                    self._apply_label_color_to_shape(napari_idx)
                    
                    # Convert from sam2_result to user_roi
                    self.shape_id_to_type[shape_id] = 'user_roi'
                    self.shape_id_to_object[shape_id] = 0  # User ROI object ID
                    # Remove session ID since it's now a regular user ROI
                    if shape_id in self.shape_id_to_session:
                        del self.shape_id_to_session[shape_id]
                    
                    print(f"✅ Finalized SAM2 result (shape ID {shape_id}) as labeled user ROI")
                
                # Update annotations: convert sam2_result to user_roi with current label
                self._convert_sam2_results_to_user_rois(current_session)
                
            else:
                print("ℹ️ No SAM2 results found to finalize")
            
            # REMOVE WORKING AREA: Remove working ROI and box prompts
            if working_area_shapes:
                print(f"🗑️ Removing {len(working_area_shapes)} working area shapes...")
                
                try:
                    # Temporarily disconnect events
                    self.shapes_layer.events.data.disconnect(self._on_shapes_changed)
                    
                    # Use napari's proper removal method
                    working_area_shapes_set = set(working_area_shapes)
                    self.shapes_layer.selected_data = working_area_shapes_set
                    self.shapes_layer.remove_selected()
                    
                    print(f"✅ Successfully removed working area shapes")
                    
                    # Reconnect events
                    self.shapes_layer.events.data.connect(self._on_shapes_changed)
                    
                except Exception as e:
                    print(f"❌ Error during working area removal: {e}")
                    try:
                        self.shapes_layer.events.data.connect(self._on_shapes_changed)
                    except:
                        pass
                    return
                    
                # Clean up working area annotations
                self._remove_annotations_by_session(['working_roi', 'box_prompt'], current_session)
                self.synchronize_all_mappings()
                
            else:
                print("ℹ️ No working area found to remove")
        
        # Clear visual elements (points)
        self.clear_point_layers()
        
        # Reset SAM2 state
        self.current_square_roi = None
        self.current_roi_params = None
        self.sam2_working_roi_id = None
        self.sam2_mode = "idle"
        
        # Clear prompts
        self.point_annotations_by_object.clear()
        self.sam2_box_prompts_by_object.clear()
        self.active_object_ids.clear()
        
        print(f"✅ SAM2 segmentation finalized with label {self.selected_label} ({self.label_names[self.selected_label - 1]})")


    def undo_sam2_seg_button_clicked(self) -> None:
        """Handle Undo SAM2 Seg button click - removes everything from CURRENT session"""
        if (getattr(self, 'sam2_mode', 'idle') == 'idle' and 
            not hasattr(self, 'sam2_session_id') or self.sam2_session_id == 0):
            print("No SAM2 segmentation to undo.")
            return
        
        current_session = getattr(self, 'sam2_session_id', 0)
        print(f"🔄 Undoing SAM2 session #{current_session}...")
        
        # Clear visual elements (points)
        self.clear_point_layers()
        
        # FIXED: Use new shape ID system to find ALL shapes from current session
        if self.shapes_layer is not None:
            shapes_to_remove = []
            
            # Find ALL shapes from current session using shape ID system
            for shape_id, session_id in self.shape_id_to_session.items():
                if session_id == current_session:
                    shape_type = self.shape_id_to_type.get(shape_id, 'unknown')
                    napari_idx = self.shape_id_to_napari_index.get(shape_id)
                    if napari_idx is not None and napari_idx < len(self.shapes_layer.data):
                        shapes_to_remove.append(napari_idx)
                        print(f"📝 Found {shape_type} (shape ID {shape_id}) at napari index {napari_idx} for removal")
            
            if shapes_to_remove:
                print(f"🗑️ Removing {len(shapes_to_remove)} shapes from session {current_session}: {shapes_to_remove}")
                
                # FIXED: Use napari's proper removal method
                try:
                    # Temporarily disconnect events to prevent interference
                    self.shapes_layer.events.data.disconnect(self._on_shapes_changed)
                    
                    # Use napari's built-in selection and removal
                    shapes_to_remove_set = set(shapes_to_remove)
                    self.shapes_layer.selected_data = shapes_to_remove_set
                    self.shapes_layer.remove_selected()
                    
                    print(f"✅ Successfully removed {len(shapes_to_remove)} shapes using napari's remove_selected")
                    
                    # Reconnect events
                    self.shapes_layer.events.data.connect(self._on_shapes_changed)
                    
                except Exception as e:
                    print(f"❌ Error during napari shape removal: {e}")
                    try:
                        self.shapes_layer.events.data.connect(self._on_shapes_changed)
                    except:
                        pass
                    return
                
                # Remove from annotations - all SAM2 types from current session
                self._remove_annotations_by_session(
                    ['working_roi', 'box_prompt', 'sam2_result'], 
                    current_session
                )
                
                # Clean up mappings
                self.synchronize_all_mappings()
                
            else:
                print(f"ℹ️ No shapes found in session {current_session}")
                
                # Debug: Print all sessions to help diagnose
                print("🔍 Debug - All shape sessions:")
                for shape_id, session_id in self.shape_id_to_session.items():
                    shape_type = self.shape_id_to_type.get(shape_id, 'unknown')
                    print(f"  Shape ID {shape_id}: session {session_id}, type {shape_type}")
        
        # Reset CURRENT SAM2 session state
        self.current_square_roi = None
        self.current_roi_params = None
        self.sam2_working_roi_id = None
        self.sam2_mode = "idle"
        self.current_sam2_object_id = 1
        
        # Clear current session data structures
        self.point_annotations_by_object.clear()
        self.sam2_box_prompts_by_object.clear()
        self.active_object_ids.clear()
        
        # Clear current session mappings
        self.object_to_box_shape_index.clear()
        self.positive_point_to_object.clear()
        self.negative_point_to_object.clear()
        self.processed_positive_points.clear()
        self.processed_negative_points.clear()
        
        # Clear propagation tracking for current session
        self.sam2_results_cache.clear()
        
        print(f"✅ SAM2 session #{current_session} undone")


    def debug_session_mapping(self) -> None:
        """Debug method to show session mapping state"""
        print("\n" + "="*60)
        print("SESSION MAPPING DEBUG")
        print("="*60)
        
        print(f"Current SAM2 session: {getattr(self, 'sam2_session_id', 'None')}")
        print(f"SAM2 mode: {getattr(self, 'sam2_mode', 'None')}")
        
        print(f"\nShape ID to Session mapping:")
        if self.shape_id_to_session:
            for shape_id, session_id in sorted(self.shape_id_to_session.items()):
                shape_type = self.shape_id_to_type.get(shape_id, 'unknown')
                napari_idx = self.shape_id_to_napari_index.get(shape_id, 'N/A')
                print(f"  Shape ID {shape_id}: session {session_id}, type {shape_type}, napari idx {napari_idx}")
        else:
            print("  Empty")
        
        print(f"\nLegacy shape to session mapping:")
        if self.shape_to_session_mapping:
            for shape_idx, session_id in sorted(self.shape_to_session_mapping.items()):
                shape_type = self.shape_to_type_mapping.get(shape_idx, 'unknown')
                print(f"  Shape index {shape_idx}: session {session_id}, type {shape_type}")
        else:
            print("  Empty")
            
        print(f"\nTotal shapes in napari layer: {len(self.shapes_layer.data) if self.shapes_layer else 0}")
        print("="*60)


    def _remove_annotations_by_session(self, annotation_types: List[str], session_id: int) -> None:
        """Remove annotations of specific types from a specific session"""
        print(f"🔍 Removing annotations of types {annotation_types} from session {session_id}")
        
        # Remove from annotation history
        original_count = len(self.annotation_history)
        removed_records = []
        kept_records = []
        
        for record in self.annotation_history:
            should_remove = False
            
            # Check annotation type match
            if record.annotation_type in annotation_types:
                # Check session ID in metadata
                if record.metadata and record.metadata.get('session_id') == session_id:
                    should_remove = True
                # Also check if no session ID but it's a working_roi or box_prompt (legacy)
                elif record.annotation_type in ['working_roi', 'box_prompt'] and session_id == getattr(self, 'sam2_session_id', 0):
                    should_remove = True
            
            if should_remove:
                removed_records.append(record)
                print(f"  📝 Removing {record.annotation_type} from session {session_id}")
            else:
                kept_records.append(record)
        
        self.annotation_history = kept_records
        removed_from_history = len(removed_records)
        
        # Remove from per-image annotations
        removed_from_images = 0
        for img_idx in list(self.all_annotations.keys()):
            original_img_count = len(self.all_annotations[img_idx])
            kept_annotations = []
            
            for ann in self.all_annotations[img_idx]:
                should_remove = False
                
                # Check annotation type match
                if ann.get('type', '') in annotation_types:
                    # Check session ID in metadata
                    if ann.get('metadata', {}).get('session_id') == session_id:
                        should_remove = True
                    # Legacy check for working_roi and box_prompt
                    elif ann.get('type') in ['working_roi', 'box_prompt'] and session_id == getattr(self, 'sam2_session_id', 0):
                        should_remove = True
                
                if not should_remove:
                    kept_annotations.append(ann)
                else:
                    print(f"  📝 Removing {ann.get('type')} from image {img_idx}, session {session_id}")
            
            self.all_annotations[img_idx] = kept_annotations
            removed_from_images += original_img_count - len(kept_annotations)
            
            # Remove empty image entries
            if not self.all_annotations[img_idx]:
                del self.all_annotations[img_idx]
        
        if removed_from_history > 0 or removed_from_images > 0:
            print(f"🗑️ Removed {removed_from_history} annotations from history, {removed_from_images} from image records (session {session_id})")
        else:
            print(f"ℹ️ No annotations found to remove for session {session_id}")


    def _remove_annotations_by_shape_id(self, shape_id: int) -> None:
        """Remove all annotations associated with a specific shape ID"""
        print(f"🗑️ Removing annotations for shape ID {shape_id}")
        
        # Get shape info before removal
        shape_type = self.shape_id_to_type.get(shape_id)
        obj_id = self.shape_id_to_object.get(shape_id)
        
        # Handle specific cleanup by shape type
        if shape_type == 'working_roi':
            print(f"  🔧 Cleaning up working ROI (shape ID {shape_id})")
            # Reset working ROI state if this was the active working ROI
            if self.sam2_working_roi_id is not None:
                working_roi_shape_id = self.napari_index_to_shape_id.get(self.sam2_working_roi_id)
                if working_roi_shape_id == shape_id:
                    self.sam2_working_roi_id = None
                    self.current_square_roi = None
                    self.current_roi_params = None
                    print(f"  🔄 Reset working ROI state")
                    
        elif shape_type == 'box_prompt' and obj_id is not None:
            print(f"  🗑️ Cleaning up box prompt for Object {obj_id}")
            if obj_id in self.sam2_box_prompts_by_object:
                del self.sam2_box_prompts_by_object[obj_id]
            if obj_id in self.object_to_box_shape_index:
                del self.object_to_box_shape_index[obj_id]
            
            # Check if object should remain active (has other annotations)
            has_points = (obj_id in self.point_annotations_by_object and 
                        len(self.point_annotations_by_object[obj_id]) > 0)
            if not has_points:
                self.active_object_ids.discard(obj_id)
                print(f"  🔄 Object {obj_id} no longer active (no annotations)")
        
        # Remove from annotation history
        original_history_count = len(self.annotation_history)
        new_history = []
        removed_count = 0
        
        for record in self.annotation_history:
            # Check if this record belongs to the shape being removed
            if record.metadata and record.metadata.get('shape_id') == shape_id:
                removed_count += 1
                print(f"  Removing {record.annotation_type} annotation from history")
            else:
                new_history.append(record)
        
        self.annotation_history = new_history
        
        # Remove from per-image annotations
        removed_from_images = 0
        for img_idx in list(self.all_annotations.keys()):
            original_img_count = len(self.all_annotations[img_idx])
            new_annotations = []
            
            for ann in self.all_annotations[img_idx]:
                # Check both in metadata and at top level
                shape_id_in_metadata = ann.get('metadata', {}).get('shape_id')
                shape_id_top_level = ann.get('shape_id')
                
                if shape_id_in_metadata == shape_id or shape_id_top_level == shape_id:
                    removed_from_images += 1
                    print(f"  Removing {ann.get('type')} annotation from image {img_idx}")
                else:
                    new_annotations.append(ann)
            
            self.all_annotations[img_idx] = new_annotations
            
            # Remove empty image entries
            if not self.all_annotations[img_idx]:
                del self.all_annotations[img_idx]
        
        print(f"  ✅ Removed {removed_count} from history, {removed_from_images} from image records")


    def _cleanup_shape_mappings_after_removal(self, removed_indices: List[int]) -> None:
        """Clean up shape mappings after shapes are removed"""
        if not removed_indices:
            return
        
        # Sort in reverse order for proper reindexing
        sorted_removed = sorted(removed_indices, reverse=True)
        
        # Remove the deleted indices from mappings
        for idx in sorted_removed:
            if idx in self.shape_to_object_mapping:
                del self.shape_to_object_mapping[idx]
            if idx in self.shape_to_type_mapping:
                del self.shape_to_type_mapping[idx]
            if idx in self.shape_to_session_mapping:
                del self.shape_to_session_mapping[idx]
        
        # Reindex remaining mappings
        new_shape_to_object = {}
        new_shape_to_type = {}
        new_shape_to_session = {}
        
        for old_idx in sorted(self.shape_to_object_mapping.keys()):
            new_idx = old_idx
            for removed_idx in sorted_removed:
                if old_idx > removed_idx:
                    new_idx -= 1
            new_shape_to_object[new_idx] = self.shape_to_object_mapping[old_idx]
            new_shape_to_type[new_idx] = self.shape_to_type_mapping[old_idx]
            if old_idx in self.shape_to_session_mapping:
                new_shape_to_session[new_idx] = self.shape_to_session_mapping[old_idx]
        
        self.shape_to_object_mapping = new_shape_to_object
        self.shape_to_type_mapping = new_shape_to_type
        self.shape_to_session_mapping = new_shape_to_session
        
        # Update working ROI ID if it still exists
        if self.sam2_working_roi_id is not None:
            if self.sam2_working_roi_id in removed_indices:
                self.sam2_working_roi_id = None
            else:
                new_working_id = self.sam2_working_roi_id
                for removed_idx in sorted_removed:
                    if self.sam2_working_roi_id > removed_idx:
                        new_working_id -= 1
                self.sam2_working_roi_id = new_working_id
    
    # ============================================================================
    # PROJECT MANAGEMENT
    # ============================================================================

    def update_viewer(self) -> None:
        """Update napari viewer"""
        if self.viewer and self.image_stack is not None:
            if self.image_layer is None:
                layer = self.viewer.add_image(
                    self.image_stack,
                    name="Images",
                    rgb=True
                )
                self.image_layer = layer[0] if isinstance(layer, list) else layer
            else:
                setattr(self.image_layer, 'data', self.image_stack)
            
            # Ensure shapes layer exists
            if self.shapes_layer is None:
                self.initialize_shapes_layer()
            
            # synchronize mappings
            self.synchronize_all_mappings()

            # Fix any dimensionality issues
            self.ensure_all_shapes_3d()
            
            # Initialize point layers
            self.initialize_point_layers()


def create_gui(pipeline: AnnotationPipeline) -> Container:
    """Create the GUI using magicgui"""
    
    # Store widget references
    widget_refs = {}
    
    # Store total images count in a proper container
    image_info = {'total_images': 0}
    
    # Create range selection widgets
    start_idx_spin = SpinBox(
        value=0,
        min=0,
        max=9999,
        name="Start Image"
    )
    
    end_idx_spin = SpinBox(
        value=10,
        min=1,
        max=10000,
        name="End Image"
    )
    
    # Label to show preview info
    preview_label = LabelWidget(value="Select folder to preview available images")
    
    # Load all checkbox
    load_all_checkbox = CheckBox(value=True, text="Load all images")
    
    # Container for range controls
    range_controls = Container(
        widgets=[start_idx_spin, end_idx_spin],
        layout='horizontal'
    )
    range_controls.visible = False
    

    # Function to update range visibility
    def toggle_range_controls(load_all: bool):
        range_controls.visible = not load_all
        if load_all:
            preview_label.value = f"Will load all {image_info['total_images']} images"
        else:
            update_preview_label()
    

    def debug_session_mapping():
        """Show detailed session mapping for debugging"""
        pipeline.debug_session_mapping()

    debug_session_btn = PushButton(text="Debug Sessions")
    debug_session_btn.clicked.connect(debug_session_mapping)


    # Function to update preview label
    def update_preview_label():
        if image_info['total_images'] > 0:
            total = image_info['total_images']
            start = start_idx_spin.value
            end = end_idx_spin.value
            count = end - start
            preview_label.value = f"Will load images {start} to {end-1} ({count} of {total} total)"
    

    # Debug SAM2 state button
    def debug_sam2_state():
        """Show detailed SAM2 state for debugging"""
        pipeline.debug_sam2_state()

    debug_sam2_btn = PushButton(text="Debug SAM2 State")
    debug_sam2_btn.clicked.connect(debug_sam2_state)


    # Connect checkbox
    load_all_checkbox.changed.connect(toggle_range_controls)
    
    # Connect spinbox changes
    start_idx_spin.changed.connect(lambda v: update_preview_label())
    end_idx_spin.changed.connect(lambda v: update_preview_label())
    
    # Folder selection with preview
    @magicgui(
        call_button="Select Folder",
        image_folder={"mode": "d"}
    )
    def select_folder(image_folder: Path = Path(".")):
        """Select folder and preview available images"""
        try:
            select_folder.img_folder = image_folder
            
            # Preview to get image count
            total_images, filenames = pipeline.preview_folders(str(image_folder))
            
            if total_images == 0:
                preview_label.value = "No images found in selected folder!"
                return
            
            # Store total for later use
            image_info['total_images'] = total_images
            
            # Update spinbox ranges
            start_idx_spin.max = total_images - 1
            end_idx_spin.max = total_images
            end_idx_spin.value = min(total_images, 10)
            
            # Update preview
            if load_all_checkbox.value:
                preview_label.value = f"Found {total_images} images. Ready to load all."
            else:
                update_preview_label()
                
            # Show first few filenames
            preview_files = filenames[:3]
            if len(filenames) > 3:
                preview_files.append("...")
            print(f"Found images: {', '.join(preview_files)}")
            
        except Exception as e:
            preview_label.value = f"Error: {e}"
    
    # Load button
    def load_images():
        """Load the selected range of images"""
        if not hasattr(select_folder, 'img_folder'):
            print("Please select folder first!")
            return
            
        try:
            # Determine range
            if load_all_checkbox.value:
                start = None
                end = None
                print("Loading all images...")
            else:
                start = start_idx_spin.value
                end = end_idx_spin.value
                print(f"Loading images {start} to {end-1}...")
            
            # Load images
            pipeline.load_image_folder(
                str(select_folder.img_folder),
                start_idx=start,
                end_idx=end
            )
            
            if pipeline.viewer:
                update_viewer()
                
            preview_label.value = f"✅ Loaded {len(pipeline.image_stack)} images successfully!"
            
        except Exception as e:
            preview_label.value = f"Error loading: {e}"
            print(f"Error loading images: {e}")
    
    load_btn = PushButton(text="Load Images")
    load_btn.clicked.connect(load_images)
    
    # Create folder section
    folder_section = Container(
        widgets=[
            select_folder,
            preview_label,
            load_all_checkbox,
            range_controls,
            load_btn
        ],
        layout='vertical'
    )
    
    # Label selection - organized in 4 columns
    label_radio_buttons = []
    
    def update_selected_label(label_num: int):
        """Update the selected label"""
        pipeline.selected_label = label_num
        # Update all radio buttons to reflect selection
        for i, rb in enumerate(label_radio_buttons):
            rb.value = (i + 1 == label_num)
        print(f"Selected label: {label_num} - {pipeline.label_names[label_num - 1]}")
    
    # Add function to handle shape selection (removed auto-conversion)
    def handle_shape_selection():
        """Handle shape selection events - for future extensibility"""
        # Removed automatic rectangle conversion to fix SAM2 issue
        # Rectangles now stay untracked until explicitly claimed by SAM2 or Convert button
        pass

    # Create radio buttons for each label
    for i, name in enumerate(pipeline.label_names):
        # Use CheckBox as radio button (only one selected at a time)
        checkbox = CheckBox(value=(i + 1 == 1))  # Default to label 1
        checkbox.text = f"{i+1}: {name}"
        
        # Create closure for checkbox connection
        def make_callback(idx):
            def callback(val):
                if val:  # Only respond to selection, not deselection
                    update_selected_label(idx)
            return callback
        
        checkbox.changed.connect(make_callback(i + 1))
        label_radio_buttons.append(checkbox)

    # FLEXIBLE: Organize labels into columns dynamically
    # Calculate optimal number of columns based on number of labels
    labels_per_column = 5  # Keep 5 labels per column as standard
    num_columns = max(1, (pipeline.num_labels + labels_per_column - 1) // labels_per_column)
    
    # Create columns
    label_columns = []
    for col in range(num_columns):
        start_idx = col * labels_per_column
        end_idx = min(start_idx + labels_per_column, pipeline.num_labels)
        
        if start_idx < end_idx:
            column_labels = label_radio_buttons[start_idx:end_idx]
            column_container = Container(widgets=column_labels, layout='vertical')
            label_columns.append(column_container)
    
    # Create horizontal container to hold all columns
    labels_layout = Container(
        widgets=label_columns,
        layout='horizontal'
    )

    # Convert rectangles to ROIs button
    def convert_rectangles():
        """Convert all untracked rectangles to user ROIs with current label"""
        if pipeline.shapes_layer is None:
            print("No shapes layer available")
            return
            
        # synchronize mappings
        pipeline.synchronize_all_mappings()

        converted_count = 0
        for idx, shape in enumerate(pipeline.shapes_layer.data):
            if (len(shape) == 4 and 
                idx not in pipeline.shape_to_object_mapping and 
                idx != pipeline.sam2_working_roi_id):
                pipeline.convert_rectangle_to_user_roi(idx)
                converted_count += 1
        
        if converted_count == 0:
            print("No untracked rectangles found to convert")
        else:
            print(f"Converted {converted_count} rectangles to manual ROIs with current label")
            update_viewer()
    
    convert_btn = PushButton(text="Convert Unused Rectangles to ROIs")
    convert_btn.clicked.connect(convert_rectangles)

    # Multi-image settings
    multi_image_spin = SpinBox(
        value=1,
        min=1,
        max=100,
        name="Propagate N images"
    )
    
    # SAM2 buttons
    def sam2_seg():
        pipeline.sam2_seg_button_clicked()
        update_viewer()
    
    sam2_seg_btn = PushButton(text="SAM2 Seg")
    sam2_seg_btn.clicked.connect(sam2_seg)
    
    # SAM2 Object ID SpinBox
    sam2_object_spin = SpinBox(
        value=1,
        min=1,
        max=99,
        name="SAM2 Object ID"
    )
    
    def update_sam2_object_id(value):
        pipeline.current_sam2_object_id = value
        print(f"\n🎯 SAM2 Object ID changed to {value}")
        print(pipeline.get_object_summary(value))
        
        if pipeline.active_object_ids:
            print(f"\nActive objects: {sorted(pipeline.active_object_ids)}")
        
        pipeline.validate_annotation_state()
    
    sam2_object_spin.changed.connect(update_sam2_object_id)
    
    # Propagate buttons
    def propagate_forward():
        num_images = multi_image_spin.value
        pipeline.propagate_button_clicked(num_images, direction="forward")
        update_viewer()

    def propagate_backward():
        num_images = multi_image_spin.value
        pipeline.propagate_button_clicked(num_images, direction="backward")
        update_viewer()

    propagate_forward_btn = PushButton(text="→ Forward")
    propagate_forward_btn.clicked.connect(propagate_forward)

    propagate_backward_btn = PushButton(text="← Backward")
    propagate_backward_btn.clicked.connect(propagate_backward)

    propagate_buttons = Container(
        widgets=[propagate_backward_btn, propagate_forward_btn],
        layout='horizontal'
    )
    
    # Refine, Delete, and Undo buttons
    def refine_seg():
        pipeline.refine_seg_button_clicked()
        update_viewer()
    
    refine_seg_btn = PushButton(text="Refine Seg")
    refine_seg_btn.clicked.connect(refine_seg)
    
    def finalize_sam2_seg():
        pipeline.finalize_sam2_seg_button_clicked()
        update_viewer()
    
    finalize_sam2_seg_btn = PushButton(text="Finalize SAM2 Seg")
    finalize_sam2_seg_btn.clicked.connect(finalize_sam2_seg)
    
    def undo_sam2_seg():
        pipeline.undo_sam2_seg_button_clicked()
        update_viewer()
    
    undo_sam2_seg_btn = PushButton(text="Undo SAM2 Seg")
    undo_sam2_seg_btn.clicked.connect(undo_sam2_seg)
    
    # Container for SAM2 control buttons
    sam2_control_buttons = Container(
        widgets=[refine_seg_btn, finalize_sam2_seg_btn, undo_sam2_seg_btn],
        layout='horizontal'
    )
    
    # Save button
    @magicgui(call_button="Save Annotations", directory={"mode": "d"})
    def save_project(directory: Path = Path(".")):
        pipeline.save_project(str(directory))
    
    # Viewer update function
    def update_viewer():
        if pipeline.viewer and pipeline.image_stack is not None:
            pipeline.update_viewer()
            # Removed automatic shape selection handler that was causing SAM2 issues
    

    def show_annotation_summary():
        """Display summary of annotations with proper categorization"""
        total_annotations = len(pipeline.annotation_history)
        annotated_images = len(pipeline.all_annotations)
        
        type_counts = {}
        label_counts = {}
        sam2_counts = {}
        
        for record in pipeline.annotation_history:
            ann_type = record.annotation_type
            
            # Categorize properly
            if ann_type in ['working_roi', 'box_prompt']:
                # Internal SAM2 state - don't count as user annotations
                sam2_counts[ann_type] = sam2_counts.get(ann_type, 0) + 1
            elif ann_type == 'sam2_result':
                # SAM2 results - count separately
                sam2_counts[ann_type] = sam2_counts.get(ann_type, 0) + 1
            else:
                # User annotations
                type_counts[ann_type] = type_counts.get(ann_type, 0) + 1
            
            # Count labels for user annotations and sam2 results
            if ann_type in ['user_roi', 'sam2_result'] and record.metadata and 'label' in record.metadata:
                label = record.metadata['label']
                # FLEXIBLE: Use num_labels for validation
                label_name = pipeline.label_names[label - 1] if 1 <= label <= pipeline.num_labels else 'unknown'
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        print(f"\n📊 ANNOTATION SUMMARY:")
        print(f"Total annotation records: {total_annotations}")
        print(f"Annotated images: {annotated_images}")
        
        print("\n🎯 User Annotations:")
        if type_counts:
            for ann_type, count in type_counts.items():
                print(f"  {ann_type}: {count}")
        else:
            print("  None")
        
        print("\n🤖 SAM2 Annotations:")
        if sam2_counts:
            for ann_type, count in sam2_counts.items():
                print(f"  {ann_type}: {count}")
        else:
            print("  None")
        
        print("\n🏷️ Label Distribution (User + SAM2 Results):")
        if label_counts:
            for label_name, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {label_name}: {count}")
        else:
            print("  None")
    

    summary_btn = PushButton(text="Show Summary")
    summary_btn.clicked.connect(show_annotation_summary)
    

    # Assemble GUI
    container = Container(widgets=[
        LabelWidget(value="=== IMAGE LOADING ==="),
        folder_section,
        LabelWidget(value="=== LABEL SELECTION ==="),
        labels_layout,
        LabelWidget(value="=== MANUAL ANNOTATION ==="),
        convert_btn,
        LabelWidget(value="=== SAM2 ANNOTATION ==="),
        sam2_seg_btn,
        sam2_object_spin,
        multi_image_spin,
        propagate_buttons,
        sam2_control_buttons,
        LabelWidget(value="=== ANNOTATION MANAGEMENT ==="),
        summary_btn,
        debug_sam2_btn,
        debug_session_btn,  # NEW: Session debug button
        save_project
    ])
    
    # Store reference to update function
    widget_refs['update_viewer'] = update_viewer
    
    # Set container properties for scrolling
    try:
        container_widget = container.native if hasattr(container, 'native') else container
        if hasattr(container_widget, 'setSizePolicy'):
            from qtpy.QtWidgets import QSizePolicy
            container_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        if hasattr(container_widget, 'setMinimumWidth'):
            container_widget.setMinimumWidth(320)
    except:
        pass
    
    return container


def main():
    """Main entry point"""
    try:
        # Create pipeline instance
        pipeline = AnnotationPipeline()
        
        # Get screen dimensions
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            
            margin_width = 400
            margin_height = 150
            
            viewer_width = min(1400, screen_width - margin_width)
            viewer_height = min(900, screen_height - margin_height)
            
            print(f"🖥️ Screen: {screen_width}x{screen_height}, Setting viewer: {viewer_width}x{viewer_height}")
            
        except Exception as e:
            print(f"Could not detect screen size: {e}, using default size")
            viewer_width, viewer_height = 1200, 800
        
        # Create napari viewer
        viewer = napari.Viewer(
            title="Annotation Pipeline with SAM2 Integration",
            show=False
        )
        
        # Set window size
        viewer.window.qt_viewer.resize(viewer_width, viewer_height)
        
        # Center the window
        try:
            screen_geometry = viewer.window.qt_viewer.screen().geometry()
            x = (screen_geometry.width() - viewer_width) // 2
            y = (screen_geometry.height() - viewer_height) // 2
            viewer.window.qt_viewer.move(x, y)
        except:
            pass
        
        pipeline.viewer = viewer
        
        # Create and add GUI
        gui = create_gui(pipeline)
        dock_widget = viewer.window.add_dock_widget(
            gui, 
            area="right", 
            name="Annotation Tools"
        )
        
        # Make dock widget scrollable
        try:
            from qtpy.QtWidgets import QScrollArea, QSizePolicy, QApplication
            from qtpy.QtCore import Qt
            
            original_widget = dock_widget.widget()
            original_widget.adjustSize()
            original_widget.updateGeometry()
            
            min_size = original_widget.minimumSizeHint()
            if not min_size.isValid():
                min_size = original_widget.sizeHint()
            
            scroll_area = QScrollArea()
            scroll_area.setWidget(original_widget)
            scroll_area.setWidgetResizable(True)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            scroll_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            original_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            
            scroll_area.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            scroll_area.setMinimumWidth(350)
            scroll_area.setMaximumWidth(450)
            
            dock_widget.setWidget(scroll_area)
            
            print("✅ Implemented scrollable dock widget")
            
        except Exception as e:
            print(f"Could not implement scrollable dock widget: {e}")
        
        # Show viewer
        viewer.show()
        
        # Initialize layers
        pipeline.initialize_point_layers()
        
        # Keyboard shortcuts
        @viewer.bind_key("s")
        def sam2_seg_key(viewer):
            """SAM2 Seg shortcut"""
            try:
                pipeline.sam2_seg_button_clicked()
            except Exception as e:
                print(f"Error in SAM2 seg: {e}")
        
        @viewer.bind_key("p")
        def propagate_key(viewer):
            """Propagate shortcut"""
            try:
                pipeline.propagate_button_clicked(1)
            except Exception as e:
                print(f"Error in propagate: {e}")
        
        @viewer.bind_key("r")
        def refine_seg_key(viewer):
            """Refine Seg shortcut"""
            try:
                pipeline.refine_seg_button_clicked()
            except Exception as e:
                print(f"Error in refine seg: {e}")
        
        @viewer.bind_key("d")
        def delete_seg_key(viewer):
            """Delete Seg shortcut"""
            try:
                pipeline.finalize_sam2_seg_button_clicked()
            except Exception as e:
                print(f"Error in delete seg: {e}")
        
        @viewer.bind_key("u")
        def undo_sam2_seg_key(viewer):
            """Undo SAM2 Seg shortcut"""
            try:
                pipeline.undo_sam2_seg_button_clicked()
            except Exception as e:
                print(f"Error in undo SAM2 seg: {e}")
        
        # Handle dimension changes
        def on_dim_change(event):
            try:
                if pipeline.image_stack is not None:
                    if hasattr(viewer.dims, 'current_step') and len(viewer.dims.current_step) > 0:
                        old_index = pipeline.current_index
                        pipeline.current_index = viewer.dims.current_step[0]
                        if old_index != pipeline.current_index:
                            print(f"Current image: {pipeline.current_index + 1}/{len(pipeline.image_stack)}")
            except Exception as e:
                print(f"Error in dimension change: {e}")
        
        viewer.dims.events.current_step.connect(on_dim_change)
        
        # Print instructions
        print("\n=== ANNOTATION PIPELINE WITH SAM2 INTEGRATION ===")
        print("🚀 Annotation-focused pipeline with SAM2 support!")
        print("\n📋 KEY FEATURES:")
        print("  ✅ All annotations are automatically saved")
        print("  ✅ SAM2 works alongside existing ROIs")
        print("  ✅ Multi-object annotation support")
        print("  ✅ Comprehensive annotation history")
        print("  ✅ Label-based color coding")
        print("\nKEYBOARD SHORTCUTS:")
        print("  s - SAM2 Seg (activate SAM2 on selected rectangle)")
        print("  p - Propagate (batch SAM2 across images)")
        print("  r - Refine Seg (enable refinement mode)")
        print("  d - Delete Seg (clear SAM2 state)")
        print("\n=== ANNOTATION WORKFLOW ===")
        print("1. LABEL SELECTION:")
        print("   - Click checkboxes to select anatomical labels")
        print("   - Selected label determines ROI color")
        print("\n2. MANUAL ANNOTATIONS:")
        print("   - Draw polygons for manual annotation (auto-colored by label)")
        print("   - Draw rectangles that remain untracked (available for SAM2)")
        print("\n3. SAM2-ASSISTED ANNOTATION:")
        print("   - Draw a rectangle anywhere in the image")
        print("   - Click 'SAM2 Seg' to claim that rectangle for SAM2 (turns green)")
        print("   - Add points/box prompts inside the green square")
        print("   - Click 'Propagate' to generate annotations")
        print("\n4. CONVERT RECTANGLES (Optional):")
        print("   - Use 'Convert Rectangles to ROIs' to turn unused rectangles into manual annotations")
        print("   - This applies the selected label color and saves them as annotations")
        print("\n5. MULTI-OBJECT ANNOTATION:")
        print("   - Change 'SAM2 Object ID' to annotate multiple objects")
        print("   - Each object gets a different color")
        print("   - All objects are processed together")
        print("\n6. SAVE ANNOTATIONS:")
        print("   - Click 'Save Annotations' to export all annotations")
        print("   - Annotations are saved as JSON files")
        print("   - Complete history and statistics included")
        
        # Start the application
        napari.run()
        
    except Exception as e:
        print(f"Critical error in main: {e}")
        import traceback
        traceback.print_exc()
        print("Please restart the application")


if __name__ == "__main__":
    main()