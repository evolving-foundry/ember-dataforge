"""
Test module - DataLoaderFactory class tests
"""
import pytest
import os
import json
import tempfile

# Try to import ray
try:
    import ray
    HAS_RAY = True
except ImportError:
    ray = None
    HAS_RAY = False
    pytestmark = pytest.mark.skipif(not HAS_RAY, reason="Ray is not installed")

from curator.utils.io.data_loader_factory import DataLoaderFactory


class TestDataLoaderFactoryDetectModality:
    """Test detect_modality method"""

    def test_detect_image_modality(self):
        """Test detecting image modality"""
        assert DataLoaderFactory.detect_modality("image.jpg") == "image"
        assert DataLoaderFactory.detect_modality("photo.png") == "image"
        assert DataLoaderFactory.detect_modality("picture.jpeg") == "image"
        assert DataLoaderFactory.detect_modality("photo.webp") == "image"
        assert DataLoaderFactory.detect_modality("image.bmp") == "image"
        assert DataLoaderFactory.detect_modality("photo.tif") == "image"

    def test_detect_audio_modality(self):
        """Test detecting audio modality"""
        assert DataLoaderFactory.detect_modality("sound.mp3") == "audio"
        assert DataLoaderFactory.detect_modality("music.wav") == "audio"
        assert DataLoaderFactory.detect_modality("audio.flac") == "audio"
        assert DataLoaderFactory.detect_modality("sound.aac") == "audio"

    def test_detect_video_modality(self):
        """Test detecting video modality"""
        assert DataLoaderFactory.detect_modality("video.mp4") == "video"
        assert DataLoaderFactory.detect_modality("movie.avi") == "video"
        assert DataLoaderFactory.detect_modality("clip.mov") == "video"
        assert DataLoaderFactory.detect_modality("film.mkv") == "video"
        assert DataLoaderFactory.detect_modality("video.webm") == "video"

    def test_detect_text_modality(self):
        """Test detecting text modality"""
        assert DataLoaderFactory.detect_modality("document.txt") == "text"
        assert DataLoaderFactory.detect_modality("data.json") == "text"
        assert DataLoaderFactory.detect_modality("records.jsonl") == "text"
        assert DataLoaderFactory.detect_modality("file.csv") == "text"
        assert DataLoaderFactory.detect_modality("data.tsv") == "text"
        assert DataLoaderFactory.detect_modality("doc.xml") == "text"
        assert DataLoaderFactory.detect_modality("readme.md") == "text"

    def test_detect_unknown_modality(self):
        """Test detecting unknown modality"""
        assert DataLoaderFactory.detect_modality("file.xyz") == "unknown"
        assert DataLoaderFactory.detect_modality("data") == "unknown"
        assert DataLoaderFactory.detect_modality(".hidden") == "unknown"

    def test_detect_case_insensitive(self):
        """Test case insensitivity"""
        assert DataLoaderFactory.detect_modality("IMAGE.JPG") == "image"
        assert DataLoaderFactory.detect_modality("Audio.MP3") == "audio"
        assert DataLoaderFactory.detect_modality("Video.MP4") == "video"


class TestDataLoaderFactoryGetReaderForModality:
    """Test get_reader_for_modality method"""

    def test_get_reader_for_image(self):
        """Test getting the image reader"""
        reader = DataLoaderFactory.get_reader_for_modality("image")
        assert reader is not None
        assert callable(reader)

    def test_get_reader_for_audio(self):
        """Test getting the audio reader"""
        reader = DataLoaderFactory.get_reader_for_modality("audio")
        assert reader is not None
        assert callable(reader)

    def test_get_reader_for_video(self):
        """Test getting the video reader"""
        reader = DataLoaderFactory.get_reader_for_modality("video")
        assert reader is not None
        assert callable(reader)

    def test_get_reader_for_text(self):
        """Test getting the text reader"""
        reader = DataLoaderFactory.get_reader_for_modality("text")
        assert reader is not None
        assert callable(reader)

    def test_get_reader_for_unsupported_modality_raises_error(self):
        """Test that unsupported modalities raise"""
        with pytest.raises(ValueError, match="Unsupported"):
            DataLoaderFactory.get_reader_for_modality("unknown_modality")


class TestDataLoaderFactoryLoadDataset:
    """Test load_dataset method"""

    def test_load_single_json_file(self, sample_json_path):
        """Test loading a single JSON file"""
        dataset = DataLoaderFactory.load_dataset(sample_json_path)
        assert dataset is not None
        assert dataset.count() == 3

    def test_load_jsonl_file(self, sample_jsonl_path):
        """Test loading a JSONL file"""
        dataset = DataLoaderFactory.load_dataset(sample_jsonl_path)
        assert dataset is not None
        assert dataset.count() == 3

    def test_load_csv_file(self, sample_csv_path):
        """Test loading a CSV file"""
        dataset = DataLoaderFactory.load_dataset(sample_csv_path)
        assert dataset is not None
        assert dataset.count() == 3

    def test_load_multiple_paths_as_list(self, sample_json_path, sample_csv_path):
        """Test loading multiple paths"""
        dataset = DataLoaderFactory.load_dataset([sample_json_path, sample_csv_path])
        assert dataset is not None

    def test_load_empty_paths_raises_error(self):
        """Test that empty paths raise"""
        with pytest.raises(Exception):
            DataLoaderFactory.load_dataset([])

    def test_load_no_valid_paths_raises_error(self):
        """Test that no valid paths raises"""
        with pytest.raises(ValueError, match="No valid"):
            DataLoaderFactory.load_dataset(["/nonexistent/file.xyz"])


class TestDataLoaderFactoryCreateMultimodalLoader:
    """Test create_multimodal_loader method"""

    def test_create_multimodal_loader(self, sample_json_path):
        """Test creating a multimodal loader"""
        config = {"data_paths": [sample_json_path]}
        loader = DataLoaderFactory.create_multimodal_loader(config)
        
        assert callable(loader)
        dataset = loader()
        assert dataset is not None


class TestDataLoaderFactoryStaticMethods:
    """Test static methods"""

    def test_read_json(self, sample_json_path):
        """Test reading a JSON file"""
        dataset = DataLoaderFactory._read_text_or_structured(sample_json_path)
        assert dataset is not None

    def test_read_text_or_structured_jsonl(self, sample_jsonl_path):
        """Test reading a JSONL file"""
        dataset = DataLoaderFactory._read_text_or_structured(sample_jsonl_path)
        assert dataset is not None

    def test_read_text_or_structured_csv(self, sample_csv_path):
        """Test reading a CSV file"""
        dataset = DataLoaderFactory._read_text_or_structured(sample_csv_path)
        assert dataset is not None

    def test_fallback_reader(self):
        """Test fallback reader"""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            dataset = DataLoaderFactory._fallback_reader(temp_path)
            assert dataset is not None
        finally:
            os.unlink(temp_path)
