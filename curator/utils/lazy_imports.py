"""
On-demand loading utility module

Provides lazy loading for external dependencies, avoiding immediate loading of heavy dependencies at module import time.
"""
import threading
from typing import Optional, Any


class LazyNLPBackend:
    """NLP backend lazy loader
    
    Load NLP libraries such as nltk and jieba on demand, avoiding import-time performance overhead.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Cached modules
        self._nltk: Optional[Any] = None
        self._jieba: Optional[Any] = None
        self._wordnet: Optional[Any] = None
        self._nltk_ngrams: Optional[Any] = None
        self._stopwords: Optional[Any] = None
        self._pos_tag: Optional[Any] = None
        self._word_tokenize: Optional[Any] = None
        self._averaged_perceptron_tagger: Optional[Any] = None
        self._averaged_perceptron_tagger_eng: Optional[Any] = None
        
        # PorterStemmer and WordNetLemmatizer instances
        self._porter_stemmer: Optional[Any] = None
        self._wordnet_lemmatizer: Optional[Any] = None
        
        self._initialized = True
    
    def ensure_nltk(self) -> bool:
        """Ensure nltk is loaded
        
        Returns:
            bool: Whether nltk is available
        """
        if self._nltk is None:
            try:
                import nltk
                self._nltk = nltk
            except ImportError:
                self._nltk = False
        return self._nltk is not False
    
    def ensure_jieba(self) -> bool:
        """Ensure jieba is loaded
        
        Returns:
            bool: Whether jieba is available
        """
        if self._jieba is None:
            try:
                import jieba
                self._jieba = jieba
            except ImportError:
                self._jieba = False
        return self._jieba is not False
    
    def ensure_nltk_data(self, data_name: str) -> bool:
        """Ensure nltk data is downloaded
        
        Args:
            data_name: Data package name, such as 'tokenizers/punkt'
            
        Returns:
            bool: Whether the data is available
        """
        if not self.ensure_nltk():
            return False
        
        try:
            self._nltk.data.find(data_name)
            return True
        except LookupError:
            try:
                self._nltk.download(data_name, quiet=True)
                return True
            except Exception:
                return False
    
    def get_nltk(self) -> Optional[Any]:
        """Get the nltk module
        
        Returns:
            nltk module, or None if unavailable
        """
        if self.ensure_nltk():
            return self._nltk
        return None
    
    def get_jieba(self) -> Optional[Any]:
        """Get the jieba module
        
        Returns:
            jieba module, or None if unavailable
        """
        if self.ensure_jieba():
            return self._jieba
        return None
    
    def get_wordnet(self) -> Optional[Any]:
        """Get the wordnet module
        
        Returns:
            wordnet module, or None if unavailable
        """
        if self._wordnet is None:
            if self.ensure_nltk_data('corpora/wordnet'):
                from nltk.corpus import wordnet
                self._wordnet = wordnet
            else:
                self._wordnet = False
        return self._wordnet if self._wordnet is not False else None
    
    def get_ngrams(self) -> Optional[Any]:
        """Get the ngrams function
        
        Returns:
            ngrams function, or None if unavailable
        """
        if self._nltk_ngrams is None:
            if self.ensure_nltk():
                from nltk.util import ngrams as nltk_ngrams
                self._nltk_ngrams = nltk_ngrams
            else:
                self._nltk_ngrams = False
        return self._nltk_ngrams if self._nltk_ngrams is not False else None
    
    def get_stopwords(self) -> Optional[Any]:
        """Get the stopwords module
        
        Returns:
            stopwords module, or None if unavailable
        """
        if self._stopwords is None:
            if self.ensure_nltk_data('corpora/stopwords'):
                from nltk.corpus import stopwords
                self._stopwords = stopwords
            else:
                self._stopwords = False
        return self._stopwords if self._stopwords is not False else None
    
    def get_pos_tag(self) -> Optional[Any]:
        """Get the pos_tag function
        
        Returns:
            pos_tag function, or None if unavailable
        """
        if self._pos_tag is None:
            if self.ensure_nltk():
                from nltk import pos_tag
                self._pos_tag = pos_tag
            else:
                self._pos_tag = False
        return self._pos_tag if self._pos_tag is not False else None
    
    def get_word_tokenize(self) -> Optional[Any]:
        """Get the word_tokenize function
        
        Returns:
            word_tokenize function, or None if unavailable
        """
        if self._word_tokenize is None:
            if self.ensure_nltk_data('tokenizers/punkt'):
                from nltk import word_tokenize
                self._word_tokenize = word_tokenize
            else:
                self._word_tokenize = False
        return self._word_tokenize if self._word_tokenize is not False else None
    
    def get_porter_stemmer(self) -> Optional[Any]:
        """Get a PorterStemmer instance
        
        Returns:
            PorterStemmer instance, or None if unavailable
        """
        if self._porter_stemmer is None:
            if self.ensure_nltk():
                from nltk.stem import PorterStemmer
                self._porter_stemmer = PorterStemmer()
            else:
                self._porter_stemmer = False
        return self._porter_stemmer if self._porter_stemmer is not False else None
    
    def get_wordnet_lemmatizer(self) -> Optional[Any]:
        """Get a WordNetLemmatizer instance
        
        Returns:
            WordNetLemmatizer instance, or None if unavailable
        """
        if self._wordnet_lemmatizer is None:
            if self.ensure_nltk_data('corpora/wordnet'):
                from nltk.stem import WordNetLemmatizer
                self._wordnet_lemmatizer = WordNetLemmatizer()
            else:
                self._wordnet_lemmatizer = False
        return self._wordnet_lemmatizer if self._wordnet_lemmatizer is not False else None
    
    def word_tokenize(self, text: str) -> list:
        """Tokenize (with fallback)
        
        Args:
            text: Input text
            
        Returns:
            List of tokenization results
        """
        tokenizer = self.get_word_tokenize()
        if tokenizer:
            return tokenizer(text)
        # Fallback strategy
        return text.split()
    
    def pos_tag(self, tokens: list) -> list:
        """Part-of-speech tagging (with fallback)
        
        Args:
            tokens: Token list
            
        Returns:
            List of tagging results
        """
        tagger = self.get_pos_tag()
        if tagger:
            return tagger(tokens)
        # Fallback strategy: mark all words as nouns
        return [(token, 'NN') for token in tokens]
    
    def jieba_cut(self, text: str) -> list:
        """jieba tokenization (with fallback)
        
        Args:
            text: Input text
            
        Returns:
            List of tokenization results
        """
        jieba = self.get_jieba()
        if jieba:
            return list(jieba.cut(text))
        # Fallback strategy: tokenize by character
        return list(text)


# Global singleton instance
_nlp_backend: Optional[LazyNLPBackend] = None


def get_nlp_backend() -> LazyNLPBackend:
    """Get the NLP backend instance (singleton)
    
    Returns:
        LazyNLPBackend instance
    """
    global _nlp_backend
    if _nlp_backend is None:
        _nlp_backend = LazyNLPBackend()
    return _nlp_backend


# Convenience import functions
def ensure_nltk() -> bool:
    """Ensure nltk is available"""
    return get_nlp_backend().ensure_nltk()


def ensure_jieba() -> bool:
    """Ensure jieba is available"""
    return get_nlp_backend().ensure_jieba()


def ensure_nltk_data(data_name: str) -> bool:
    """Ensure nltk data is downloaded"""
    return get_nlp_backend().ensure_nltk_data(data_name)


# Fallback utility functions
def simple_tokenize(text: str) -> list:
    """Simple tokenization (fallback)"""
    return text.lower().split()


def simple_ngrams(tokens: list, n: int) -> list:
    """Simple ngram generation (fallback)"""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def get_default_stopwords() -> set:
    """Get the default stop-word set (fallback)"""
    return {
        'the', 'is', 'at', 'which', 'that', 'and', 'or', 'a', 'an', 
        'in', 'on', 'of', 'to', 'for', 'with', 'as', 'by', 'from',
        'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'
    }




class LazyCV2Backend:
    """CV2 backend lazy loader

    Load the cv2 library on demand, avoiding import-time performance overhead.
    Mainly used by image and video processing operations.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Cache the cv2 module
        self._cv2: Optional[Any] = None

        self._initialized = True

    def ensure_cv2(self) -> bool:
        """Ensure cv2 is loaded

        Returns:
            bool: Whether cv2 is available
        """
        if self._cv2 is None:
            try:
                import cv2
                self._cv2 = cv2
            except ImportError:
                self._cv2 = False
        return self._cv2 is not False

    def get_cv2(self) -> Optional[Any]:
        """Get the cv2 module

        Returns:
            cv2 module, or None if unavailable
        """
        if self.ensure_cv2():
            return self._cv2
        return None

    def imread(self, path: str, flags: int = None) -> Optional[Any]:
        """Read an image (with fallback)

        Args:
            path: Image path
            flags: Read flags (such as cv2.IMREAD_GRAYSCALE)

        Returns:
            Image array or None
        """
        cv2 = self.get_cv2()
        if cv2:
            if flags is not None:
                return cv2.imread(path, flags)
            return cv2.imread(path)
        return None

    def imwrite(self, path: str, image: Any) -> bool:
        """Save an image

        Args:
            path: Output path
            image: Image array

        Returns:
            Whether saving succeeded
        """
        cv2 = self.get_cv2()
        if cv2:
            return cv2.imwrite(path, image)
        return False

    def Canny(self, image: Any, threshold1: int, threshold2: int) -> Optional[Any]:
        """Canny edge detection

        Args:
            image: Input image
            threshold1: Low threshold
            threshold2: High threshold

        Returns:
            Edge image or None
        """
        cv2 = self.get_cv2()
        if cv2:
            return cv2.Canny(image, threshold1, threshold2)
        return None

    def flip(self, image: Any, flipCode: int) -> Optional[Any]:
        """Flip an image

        Args:
            image: Input image
            flipCode: 0=vertical, 1=horizontal, -1=both

        Returns:
            Flipped image or None
        """
        cv2 = self.get_cv2()
        if cv2:
            return cv2.flip(image, flipCode)
        return None

    def cvtColor(self, image: Any, code: int) -> Optional[Any]:
        """Color space conversion

        Args:
            image: Input image
            code: Color conversion code (such as cv2.COLOR_GRAY2BGR)

        Returns:
            Converted image or None
        """
        cv2 = self.get_cv2()
        if cv2:
            return cv2.cvtColor(image, code)
        return None

    def VideoCapture(self, path: str) -> Optional[Any]:
        """Open a video file

        Args:
            path: Video path

        Returns:
            VideoCapture object or None
        """
        cv2 = self.get_cv2()
        if cv2:
            return cv2.VideoCapture(path)
        return None

    def VideoWriter(self, path: str, fourcc: Any, fps: float, frameSize: tuple) -> Optional[Any]:
        """Create a video writer

        Args:
            path: Output path
            fourcc: Codec
            fps: Frame rate
            frameSize: Frame size

        Returns:
            VideoWriter object or None
        """
        cv2 = self.get_cv2()
        if cv2:
            return cv2.VideoWriter(path, fourcc, fps, frameSize)
        return None

    def VideoWriter_fourcc(self, c1: str, c2: str, c3: str, c4: str) -> Optional[Any]:
        """Create a fourcc code

        Args:
            c1-c4: Four characters

        Returns:
            fourcc code or None
        """
        cv2 = self.get_cv2()
        if cv2:
            return cv2.VideoWriter_fourcc(c1, c2, c3, c4)
        return None

    def copyMakeBorder(self, image: Any, top: int, bottom: int, left: int, right: int,
                       borderType: int, value: Any = None) -> Optional[Any]:
        """Add a border

        Args:
            image: Input image
            top, bottom, left, right: Border widths
            borderType: Border type
            value: Fill value

        Returns:
            Image with border added, or None
        """
        cv2 = self.get_cv2()
        if cv2:
            if value is not None:
                return cv2.copyMakeBorder(image, top, bottom, left, right, borderType, value=value)
            return cv2.copyMakeBorder(image, top, bottom, left, right, borderType)
        return None

    def resize(self, image: Any, dsize: tuple) -> Optional[Any]:
        """Resize an image

        Args:
            image: Input image
            dsize: Target size(width, height)

        Returns:
            Resized image or None
        """
        cv2 = self.get_cv2()
        if cv2:
            return cv2.resize(image, dsize)
        return None

    # Constant access
    @property
    def IMREAD_GRAYSCALE(self) -> int:
        """Grayscale read flag"""
        cv2 = self.get_cv2()
        if cv2:
            return cv2.IMREAD_GRAYSCALE
        return 0

    @property
    def COLOR_GRAY2BGR(self) -> int:
        """Grayscale-to-BGR flag"""
        cv2 = self.get_cv2()
        if cv2:
            return cv2.COLOR_GRAY2BGR
        return 8

    @property
    def BORDER_CONSTANT(self) -> int:
        """Constant border flag"""
        cv2 = self.get_cv2()
        if cv2:
            return cv2.BORDER_CONSTANT
        return 0

    @property
    def CAP_PROP_FPS(self) -> int:
        """FPS property"""
        cv2 = self.get_cv2()
        if cv2:
            return cv2.CAP_PROP_FPS
        return 5

    @property
    def CAP_PROP_FRAME_WIDTH(self) -> int:
        """Frame width property"""
        cv2 = self.get_cv2()
        if cv2:
            return cv2.CAP_PROP_FRAME_WIDTH
        return 3

    @property
    def CAP_PROP_FRAME_HEIGHT(self) -> int:
        """Frame height property"""
        cv2 = self.get_cv2()
        if cv2:
            return cv2.CAP_PROP_FRAME_HEIGHT
        return 4

    @property
    def CAP_PROP_FRAME_COUNT(self) -> int:
        """Frame count property"""
        cv2 = self.get_cv2()
        if cv2:
            return cv2.CAP_PROP_FRAME_COUNT
        return 7


# Global CV2 singleton instance
_cv2_backend: Optional[LazyCV2Backend] = None


def get_cv2_backend() -> LazyCV2Backend:
    """Get the CV2 backend instance (singleton)

    Returns:
        LazyCV2Backend instance
    """
    global _cv2_backend
    if _cv2_backend is None:
        _cv2_backend = LazyCV2Backend()
    return _cv2_backend


def ensure_cv2() -> bool:
    """Ensure cv2 is available"""
    return get_cv2_backend().ensure_cv2()
