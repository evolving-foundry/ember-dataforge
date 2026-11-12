#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common function module

Provides shared functionality for all operator examples under examples/curator/single_ops/:
1. Ray initialization
2. Image result display
"""

import os
import sys
import time
import re
from typing import Optional, List, Tuple


def init_ray(num_cpus: int = 2, object_store_memory: int = 512 * 1024 * 1024) -> bool:
    """Initialize the Ray environment
    
    Initial step for all operator examples; configures the Ray environment consistently.
    
    Args:
        num_cpus: Number of CPU cores to use, defaults to 2
        object_store_memory: Object store memory size in bytes, defaults to 512MB
        
    Returns:
        Whether Ray was initialized successfully
    """
    print("Configuring the Ray environment...")
    os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
    os.environ['RAY_ENABLE_MAC_LARGE_OBJECT_STORE'] = '1'

    print("Initializing Ray...")
    try:
        import ray

        if ray.is_initialized():
            print("Ray is already initialized; shutting it down for reconfiguration...")
            ray.shutdown()
            time.sleep(1)

        start_time = time.time()
        ray.init(
            ignore_reinit_error=True,
            num_cpus=num_cpus,
            object_store_memory=object_store_memory,
            include_dashboard=False,
            log_to_driver=False
        )
        init_time = time.time() - start_time
        print(f"Ray initialized successfully (elapsed: {init_time:.2f}s)")
        return True

    except Exception as e:
        print(f"Ray initialization failed: {e}")
        print("Trying local mode...")
        try:
            import ray
            ray.init(
                local_mode=True,
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False
            )
            print("Ray local mode initialized successfully")
            return True
        except Exception as e2:
            print(f"Ray local mode also failed: {e2}")
            print("Ray initialization failed; continuing to run...")
            return False


def shutdown_ray():
    """Shut down the Ray environment"""
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown successfully")
    except Exception:
        pass


def display_image_results(
    output_path: str,
    image_field_name: str = 'canny_image_data',
    max_files: int = 5,
    save_images: bool = True,
    images_output_dir: Optional[str] = None,
    image_format: str = 'png',
    colormap: str = 'grayscale'
) -> None:
    """Display result information after image operator processing and save image data to files
    
    Reads Parquet files, displays image data shape information, and saves images as files.
    
    Args:
        output_path: Output directory path (directory containing the Parquet files)
        image_field_name: Image field name, defaults to 'canny_image_data'
        max_files: Maximum number of files to process, defaults to 5
        save_images: Whether to save image files, defaults to True
        images_output_dir: Image output directory, defaults to the images subdirectory under output_path
        image_format: Image format, 'png' or 'jpg', defaults to 'png'
        colormap: Color map, 'grayscale' or 'jet', defaults to 'grayscale'
    """
    print(f"\nOutput file information:")
    if not os.path.exists(output_path):
        print(f"  File exists: False")
        return
    
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size} bytes")
    print(f"  File exists: True")
    print(f"  Output path: {output_path}")
    
    # Set the image output directory
    if images_output_dir is None:
        images_output_dir = os.path.join(output_path, 'images')
    
    if save_images:
        os.makedirs(images_output_dir, exist_ok=True)
        print(f"\nImage output directory: {images_output_dir}")
    
    # Read and display shape information from the output files
    print(f"\nOutput data shape information:")
    try:
        import pyarrow.parquet as pq
        import numpy as np
        from PIL import Image
        
        parquet_files = [f for f in os.listdir(output_path) if f.endswith('.parquet')]
        saved_count = 0
        
        for i, pf in enumerate(parquet_files[:max_files]):
            pf_path = os.path.join(output_path, pf)
            table = pq.read_table(pf_path)
            schema = table.schema
            
            # Find the image field
            image_found = False
            for field_idx in range(len(schema)):
                field = schema.field(field_idx)
                if field.name == image_field_name:
                    field_type = str(field.type)
                    
                    # Case 1: Arrow Tensor Extension type (contains shape= information)
                    if 'shape=' in field_type:
                        # Fix: shape uses parentheses not brackets
                        shape_match = re.search(r'shape=\(([^)]+)\)', field_type)
                        if shape_match:
                            shape_str = shape_match.group(1)
                            print(f"\n  File {i+1}: {pf}")
                            print(f"    Image shape: ({shape_str})")
                            print(f"    Data type: uint8")
                            image_found = True
                    
                        # Read image data
                        if save_images and image_found:
                            try:
                                canny_column = table.column(field_idx)
                                # LargeListScalar requires as_py() to get the data
                                list_scalar = canny_column[0]  # Get the first row
                                
                                # Use as_py() to get a Python object, then convert to numpy
                                py_val = list_scalar.as_py()
                                
                                # Convert the list to a numpy array
                                image_data = np.array(py_val, dtype=np.uint8)
                                
                                # Ensure the data range is 0-255
                                if image_data.dtype != np.uint8:
                                    image_data = image_data.astype(np.uint8)
                                
                                # Generate the output filename
                                base_name = os.path.splitext(pf)[0]
                                output_filename = f"{image_field_name}_{i+1:03d}.{image_format}"
                                
                                output_image_path = os.path.join(images_output_dir, output_filename)
                                
                                # Save the image according to the color map
                                if len(image_data.shape) == 2:
                                    # Grayscale image
                                    if colormap == 'grayscale':
                                        image = Image.fromarray(image_data, mode='L')
                                    else:
                                        image = Image.fromarray(image_data, mode='L')
                                elif len(image_data.shape) == 3 and image_data.shape[2] == 3:
                                    # RGB color image
                                    image = Image.fromarray(image_data, mode='RGB')
                                elif len(image_data.shape) == 3 and image_data.shape[2] == 4:
                                    # RGBA image
                                    image = Image.fromarray(image_data, mode='RGBA')
                                else:
                                    # Other formats: try direct conversion
                                    image = Image.fromarray(image_data)
                                
                                image.save(output_image_path)
                                print(f"    Image saved: {output_filename}")
                                saved_count += 1

                            except Exception as img_e:
                                print(f"    Error saving image: {img_e}")
                                import traceback
                                traceback.print_exc()
                    
                    # Case 2: large_binary or ray pickled object type (list data serialized with pickle)
                    elif 'large_binary' in field_type or 'binary' in field_type or 'arrow_pickled_object' in field_type:
                        print(f"\n  File {i+1}: {pf}")
                        print(f"    Field type: binary or pickled object data")
                        image_found = True

                        if save_images and image_found:
                            try:
                                import pickle
                                
                                # Read the binary column
                                binary_column = table.column(field_idx)
                                binary_data = binary_column[0].as_py()
                                
                                # arrow_pickled_object is auto-deserialized by Ray; it may be bytes, list, or PIL Image
                                if isinstance(binary_data, bytes):
                                    # Trust boundary: this example only inspects local
                                    # Ray/Arrow data produced by the local pipeline.
                                    # Never unpickle bytes from untrusted uploads.
                                    image_list = pickle.loads(binary_data)
                                elif isinstance(binary_data, list):
                                    image_list = binary_data
                                elif hasattr(binary_data, 'save'):
                                    pil_image = binary_data
                                    output_filename = f"{image_field_name}_{i+1:03d}.{image_format}"
                                    output_image_path = os.path.join(images_output_dir, output_filename)
                                    pil_image.save(output_image_path)
                                    print(f"    PIL image saved: {output_filename}")
                                    saved_count += 1
                                    continue
                                else:
                                    print(f"    Unknown data type: {type(binary_data)}")
                                    continue
                                
                                print(f"    List contains {len(image_list)} cropped images")
                                
                                # Save each image in the list
                                for img_idx, image_data in enumerate(image_list):
                                    if not isinstance(image_data, np.ndarray):
                                        continue
                                    
                                    print(f"    Image {img_idx+1} shape: {image_data.shape}")
                                    
                                    # Ensure the data range is 0-255
                                    if image_data.dtype != np.uint8:
                                        image_data = image_data.astype(np.uint8)
                                    
                                    # Generate the output filename
                                    output_filename = f"{image_field_name}_{i+1:03d}_crop{img_idx+1}.{image_format}"
                                    output_image_path = os.path.join(images_output_dir, output_filename)
                                    
                                    # Save the image according to the number of channels
                                    if len(image_data.shape) == 2:
                                        image = Image.fromarray(image_data, mode='L')
                                    elif len(image_data.shape) == 3 and image_data.shape[2] == 3:
                                        image_data = image_data[:, :, ::-1]
                                        image = Image.fromarray(image_data, mode='RGB')
                                    elif len(image_data.shape) == 3 and image_data.shape[2] == 4:
                                        image = Image.fromarray(image_data, mode='RGBA')
                                    else:
                                        image = Image.fromarray(image_data)
                                    
                                    image.save(output_image_path)
                                    print(f"      Image saved: {output_filename}")
                                    saved_count += 1
                                    
                            except Exception as img_e:
                                print(f"    Error saving image: {img_e}")
                                import traceback
                                traceback.print_exc()
                    
                    # Case 3: large_list type (e.g. large_list<element: uint8>)
                    elif 'list' in field_type.lower() or 'large_list' in field_type:
                        print(f"\n  File {i+1}: {pf}")
                        print(f"    DEBUG: field_type = {field_type}")
                        print(f"    Field type: list data (large_list)")
                        image_found = True

                        if save_images and image_found:
                            try:
                                # Read the list column
                                list_column = table.column(field_idx)
                                list_data = list_column[0].as_py()  # Get the list data from the first row

                                if isinstance(list_data, list):
                                    # Convert the list to a numpy array
                                    image_data = np.array(list_data, dtype=np.uint8)
                                    
                                    # Try to infer the shape - requires the original image
                                    print(f"    Data length: {len(list_data)}")

                                    # Try to find the original image dimensions
                                    # Infer from the filename or use a default shape
                                    # Assume a grayscale image and try common image sizes
                                    possible_shapes = [
                                        (256, 256),
                                        (512, 512),
                                        (480, 480),
                                        (640, 480),
                                        (800, 600),
                                        (1024, 768),
                                    ]
                                    
                                    best_shape = None
                                    for shape in possible_shapes:
                                        if shape[0] * shape[1] == len(list_data):
                                            best_shape = shape
                                            break
                                    
                                    if best_shape:
                                        image_data = image_data.reshape(best_shape)
                                        print(f"    Inferred shape: {best_shape}")
                                    else:
                                        # Cannot infer shape, save as raw data
                                        print(f"    Unable to infer shape; saving raw data")
                                        image_data = image_data.reshape(-1, 1)
                                    
                                    # Generate the output filename
                                    base_name = os.path.splitext(pf)[0]
                                    output_filename = f"{image_field_name}_{i+1:03d}.{image_format}"
                                    output_image_path = os.path.join(images_output_dir, output_filename)
                                    
                                    # Save the image according to its shape
                                    if len(image_data.shape) == 2:
                                        image = Image.fromarray(image_data, mode='L')
                                        image.save(output_image_path)
                                        print(f"    Image saved: {output_filename}")
                                        saved_count += 1
                                    else:
                                        print(f"    Unable to recognize image shape: {image_data.shape}")
                                else:
                                    print(f"    Data is not a list: {type(list_data)}")
                                    
                            except Exception as img_e:
                                print(f"    Error saving image: {img_e}")
                                import traceback
                                traceback.print_exc()
                    
                    break
            
            if not image_found:
                print(f"\n  File {i+1}: {pf}")
                print(f"    Image field not found: {image_field_name}")
        
        if save_images:
            print(f"\nSaved {saved_count} image files to: {images_output_dir}")
            
    except Exception as e:
        print(f"  Error reading shape information: {e}")
        import traceback
        traceback.print_exc()


def display_video_results(
    output_path: str,
    video_field_name: str = 'video_path',
    max_files: int = 5,
    save_videos: bool = True,
    videos_output_dir: Optional[str] = None,
    video_format: str = 'mp4'
) -> None:
    """Display result information after video operator processing and save video files
    
    Reads Parquet files, displays video metadata, and copies video files to the target directory.
    
    Args:
        output_path: Output directory path (directory containing the Parquet files)
        video_field_name: Video field name, defaults to 'video_path'
        max_files: Maximum number of files to process, defaults to 5
        save_videos: Whether to save video files, defaults to True
        videos_output_dir: Video output directory, defaults to the videos subdirectory under output_path
        video_format: Video format, 'mp4' / 'avi' / 'mov' etc., defaults to 'mp4'
    """
    print(f"\nOutput file information:")
    if not os.path.exists(output_path):
        print(f"  File exists: False")
        return
    
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size} bytes")
    print(f"  File exists: True")
    print(f"  Output path: {output_path}")
    
    # Set the video output directory
    if videos_output_dir is None:
        videos_output_dir = os.path.join(output_path, 'videos')
    
    if save_videos:
        os.makedirs(videos_output_dir, exist_ok=True)
        print(f"\nVideo output directory: {videos_output_dir}")
    
    # Read and display video information from the output files
    print(f"\nOutput video information:")
    try:
        import pyarrow.parquet as pq
        import shutil

        parquet_files = [f for f in os.listdir(output_path) if f.endswith('.parquet')]
        saved_count = 0

        for i, pf in enumerate(parquet_files[:max_files]):
            pf_path = os.path.join(output_path, pf)

            # If it is a directory (Ray Data parquet output), get the parquet files inside
            if os.path.isdir(pf_path):
                inner_files = [f for f in os.listdir(pf_path) if f.endswith('.parquet')]
                if inner_files:
                    pf_path = os.path.join(pf_path, inner_files[0])
                else:
                    continue

            try:
                table = pq.read_table(pf_path)
            except Exception as read_err:
                print(f"\n  File {i+1}: {pf}")
                print(f"    Error reading parquet file: {read_err}")
                continue

            schema = table.schema
            
            # Find the video field
            video_found = False
            for field_idx in range(len(schema)):
                field = schema.field(field_idx)
                if field.name == video_field_name:
                    video_found = True
                    
                    # Read the video path
                    try:
                        video_column = table.column(field_idx)
                        video_path = str(video_column[0])  # Get the video path from the first row
                        
                        print(f"\n  File {i+1}: {pf}")
                        print(f"    Video path: {video_path}")
                        
                        # Get video metadata
                        if os.path.exists(video_path):
                            video_size = os.path.getsize(video_path)
                            print(f"    Video size: {video_size / (1024*1024):.2f} MB")
                            
                            # Use OpenCV to get video metadata
                            try:
                                import cv2
                                cap = cv2.VideoCapture(video_path)
                                if cap.isOpened():
                                    fps = cap.get(cv2.CAP_PROP_FPS)
                                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    duration = frame_count / fps if fps > 0 else 0
                                    
                                    print(f"    Resolution: {width}x{height}")
                                    print(f"    FPS: {fps:.2f}")
                                    print(f"    Frame count: {frame_count}")
                                    print(f"    Duration: {duration:.2f} seconds")
                                    cap.release()
                            except Exception as cv_e:
                                print(f"    Unable to read video metadata: {cv_e}")
                        else:
                            print(f"    Warning: video file does not exist")
                        
                        # Copy the video file
                        if save_videos and os.path.exists(video_path):
                            try:
                                # Generate the output filename
                                output_filename = f"{video_field_name}_{i+1:03d}.{video_format}"
                                output_video_path = os.path.join(videos_output_dir, output_filename)
                                
                                shutil.copy2(video_path, output_video_path)
                                print(f"    Video saved: {output_filename}")
                                saved_count += 1
                            except Exception as copy_e:
                                print(f"    Error saving video: {copy_e}")
                        
                    except Exception as e:
                        print(f"\n  File {i+1}: {pf}")
                        print(f"    Error reading video information: {e}")
                    break
            
            if not video_found:
                print(f"\n  File {i+1}: {pf}")
                print(f"    Video field not found: {video_field_name}")
        
        if save_videos:
            print(f"\nSaved {saved_count} video files to: {videos_output_dir}")
            
    except Exception as e:
        print(f"  Error reading video information: {e}")
        import traceback
        traceback.print_exc()


def display_audio_results(
    output_path: str,
    audio_field_name: str = 'audio',
    max_files: int = 5,
    save_audio: bool = True,
    audio_output_dir: Optional[str] = None,
    audio_format: str = 'mp3'
) -> None:
    """Display result information after audio operator processing and save audio files
    
    Reads Parquet files, displays audio metadata, and copies audio files to the target directory.
    
    Args:
        output_path: Output directory path (directory containing the Parquet files)
        audio_field_name: Audio field name, defaults to 'audio'
        max_files: Maximum number of files to process, defaults to 5
        save_audio: Whether to save audio files, defaults to True
        audio_output_dir: Audio output directory, defaults to the audio subdirectory under output_path
        audio_format: Audio format, 'mp3' / 'wav' / 'aac' etc., defaults to 'mp3'
    """
    print(f"\nOutput file information:")
    if not os.path.exists(output_path):
        print(f"  File exists: False")
        return
    
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size} bytes")
    print(f"  File exists: True")
    print(f"  Output path: {output_path}")
    
    # Set the audio output directory
    if audio_output_dir is None:
        audio_output_dir = os.path.join(output_path, 'audio')
    
    if save_audio:
        os.makedirs(audio_output_dir, exist_ok=True)
        print(f"\nAudio output directory: {audio_output_dir}")
    
    # Read and display audio information from the output files
    print(f"\nOutput audio information:")
    try:
        import pyarrow.parquet as pq
        import shutil
        
        parquet_files = [f for f in os.listdir(output_path) if f.endswith('.parquet')]
        saved_count = 0
        
        for i, pf in enumerate(parquet_files[:max_files]):
            pf_path = os.path.join(output_path, pf)
            table = pq.read_table(pf_path)
            schema = table.schema
            
            # Find the audio field
            audio_found = False
            for field_idx in range(len(schema)):
                field = schema.field(field_idx)
                if field.name == audio_field_name:
                    audio_found = True
                    
                    # Read the audio path
                    try:
                        audio_column = table.column(field_idx)
                        audio_path = str(audio_column[0])
                        
                        print(f"\n  File {i+1}: {pf}")
                        print(f"    Audio path: {audio_path}")
                        
                        # Get audio metadata
                        if os.path.exists(audio_path):
                            audio_size = os.path.getsize(audio_path)
                            print(f"    Audio size: {audio_size / (1024*1024):.2f} MB")
                            
                            # Use mutagen to get audio metadata
                            try:
                                from mutagen import File as MutagenFile
                                audio_info = MutagenFile(audio_path)
                                if audio_info is not None:
                                    print(f"    Duration: {audio_info.info.length:.2f} seconds")
                                    if hasattr(audio_info.info, 'bitrate'):
                                        print(f"    Bitrate: {audio_info.info.bitrate / 1000:.0f} kbps")
                                    if hasattr(audio_info.info, 'sample_rate'):
                                        print(f"    Sample rate: {audio_info.info.sample_rate} Hz")
                            except ImportError:
                                print(f"    (mutagen is not installed; detailed metadata is unavailable)")
                            except Exception as mu_e:
                                print(f"    Unable to read audio metadata: {mu_e}")
                        else:
                            print(f"    Warning: audio file does not exist")
                        
                        # Copy the audio file
                        if save_audio and os.path.exists(audio_path):
                            try:
                                output_filename = f"{audio_field_name}_{i+1:03d}.{audio_format}"
                                output_audio_path = os.path.join(audio_output_dir, output_filename)
                                shutil.copy2(audio_path, output_audio_path)
                                print(f"    Audio saved: {output_filename}")
                                saved_count += 1
                            except Exception as copy_e:
                                print(f"    Error saving audio: {copy_e}")
                        
                    except Exception as e:
                        print(f"\n  File {i+1}: {pf}")
                        print(f"    Error reading audio information: {e}")
                    break
            
            if not audio_found:
                print(f"\n  File {i+1}: {pf}")
                print(f"    Audio field not found: {audio_field_name}")
        
        if save_audio:
            print(f"\nSaved {saved_count} audio files to: {audio_output_dir}")
            
    except Exception as e:
        print(f"  Error reading audio information: {e}")
        import traceback
        traceback.print_exc()


def display_text_results(
    output_path: str,
    text_field_name: str = 'text',
    max_files: int = 5,
    save_text: bool = True,
    text_output_dir: Optional[str] = None,
    text_format: str = 'txt',
    max_text_length: int = 200
) -> None:
    """Display result information after text operator processing and save text files
    
    Reads Parquet files, displays text content summaries, and saves text to files.
    
    Args:
        output_path: Output directory path (directory containing the Parquet files)
        text_field_name: Text field name, defaults to 'text'
        max_files: Maximum number of files to process, defaults to 5
        save_text: Whether to save text files, defaults to True
        text_output_dir: Text output directory, defaults to the texts subdirectory under output_path
        text_format: Text format, 'txt' / 'json' etc., defaults to 'txt'
        max_text_length: Maximum display length for a single text entry, defaults to 200
    """
    print(f"\nOutput file information:")
    if not os.path.exists(output_path):
        print(f"  File exists: False")
        return
    
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size} bytes")
    print(f"  File exists: True")
    print(f"  Output path: {output_path}")
    
    # Set the text output directory
    if text_output_dir is None:
        text_output_dir = os.path.join(output_path, 'texts')
    
    if save_text:
        os.makedirs(text_output_dir, exist_ok=True)
        print(f"\nText output directory: {text_output_dir}")
    
    # Read and display text information from the output files
    print(f"\nOutput text information:")
    try:
        import pyarrow.parquet as pq
        
        parquet_files = [f for f in os.listdir(output_path) if f.endswith('.parquet')]
        total_records = 0
        saved_count = 0
        
        for i, pf in enumerate(parquet_files[:max_files]):
            pf_path = os.path.join(output_path, pf)
            table = pq.read_table(pf_path)
            schema = table.schema
            
            # Find the text field
            text_found = False
            for field_idx in range(len(schema)):
                field = schema.field(field_idx)
                if field.name == text_field_name:
                    text_found = True
                    
                    try:
                        text_column = table.column(field_idx)
                        num_rows = table.num_rows
                        total_records += num_rows
                        
                        print(f"\n  File {i+1}: {pf}")
                        print(f"    Record count: {num_rows}")
                        
                        # Display the first few text entries
                        for row_idx in range(min(3, num_rows)):
                            text_value = text_column[row_idx].as_py()
                            if text_value:
                                text_str = str(text_value)
                                if len(text_str) > max_text_length:
                                    text_str = text_str[:max_text_length] + "..."
                                print(f"    Text {row_idx+1}: {text_str}")
                        
                        # Save text files
                        if save_text:
                            try:
                                all_text = []
                                for row_idx in range(num_rows):
                                    text_value = text_column[row_idx].as_py()
                                    if text_value:
                                        all_text.append(str(text_value))
                                
                                if all_text:
                                    output_filename = f"{text_field_name}_{i+1:03d}.{text_format}"
                                    output_text_path = os.path.join(text_output_dir, output_filename)
                                    
                                    with open(output_text_path, 'w', encoding='utf-8') as f:
                                        f.write('\n'.join(all_text))
                                    
                                    print(f"    Text saved: {output_filename}")
                                    saved_count += 1
                            
                            except Exception as save_e:
                                print(f"    Error saving text: {save_e}")
                    
                    except Exception as e:
                        print(f"\n  File {i+1}: {pf}")
                        print(f"    Error reading text information: {e}")
                    break
            
            if not text_found:
                print(f"\n  File {i+1}: {pf}")
                print(f"    Text field not found: {text_field_name}")
        
        print(f"\nTotal: {total_records} text records")
        if save_text:
            print(f"Saved {saved_count} text files to: {text_output_dir}")
            
    except Exception as e:
        print(f"  Error reading text information: {e}")
        import traceback
        traceback.print_exc()


def get_project_root() -> str:
    """Get the project root directory
    
    Returns:
        Project root directory path
    """
    # Infer from the current file location
    current_file = os.path.abspath(__file__)
    examples_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(os.path.dirname(examples_dir))
    return project_root


def setup_path():
    """Set the Python path by adding the project root directory to sys.path"""
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.environ['PYTHONPATH'] = project_root + os.pathsep + os.environ.get('PYTHONPATH', '')
    return project_root
