# Classes to encapsulate an Image object

import os
from typing import Optional, Self
from attr import dataclass
import cv2
from numpy import ndarray

@dataclass
class LocationInfo:
    """Class to capture Location lat/long info for an image or video"""
    latitude: float
    longitude: float 
    positionAccuracy: Optional[float] = None

@dataclass
class AppImage:
    """Class representing an Image object"""
    cv2ImageBuffer: ndarray
    filePath: Optional[str] = None
    fileChecksumHex: Optional[str] = None
    faceCount: Optional[int] = None
    locationInfo: Optional[LocationInfo] = None

    # Forward references can be specified as string literals which translate to valid data types, 
    #  so using 'AppImage' as return type. 
    @staticmethod
    def from_filePath(filePath: str) -> 'Optional[AppImage]': 
        """
        Create an AppImage object from a file path, image buffer will be loaded internally.
        The filepath must be valid. 
        """
        if not os.path.exists(filePath):
            print(f"Failed to create AppImage object from filepath: {filePath}")
            return None
        return AppImage(filePath=filePath, cv2ImageBuffer=cv2.imread(filePath))
    
    @staticmethod
    def from_cv2imreadbuffer(cv2ImageBuffer: ndarray) -> 'Optional[AppImage]':
        """
        Create an AppImage object from an ndarray obtained via cv2.imread API call.
        This would be useful for holding temporary images generated while processing, cropping etc. 
        which are not saved on disk. 
        """
        if not cv2ImageBuffer:
            return None
        return AppImage(cv2ImageBuffer=cv2ImageBuffer)
    