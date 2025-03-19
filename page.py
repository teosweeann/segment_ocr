import numpy as np
import box_ops
import cv2
import ocr
from functools import lru_cache
from typing import List, Tuple, Any

Box = Tuple[int, int, int, int]  # (top, bottom, left, right)

class Page:
    """
    Represents a page image along with bounding boxes to extract regions of interest.
    Provides utility methods to crop, classify, and transcribe text from the page.
    """
    def __init__(self, page: Any, bnd_boxes: List[Box], page_hr=None) -> None:
        """
        Args:
            page: The page image as a NumPy array.
            bnd_boxes: List of bounding boxes defined as (top, bottom, left, right).
        """
        self.page = page
        self.page_hr = page_hr
        self.bnd_boxes = bnd_boxes
        self.height = page.shape[0]
        self.width = page.shape[1]

    def __getitem__(self, index: int) -> Any:
        """Returns the cropped image of the bounding box at the given index with an extension."""
        return box_ops.crop_img(self.page, self.bnd_boxes[index], ext=5)

    def __len__(self) -> int:
        """Returns the number of bounding boxes."""
        return len(self.bnd_boxes)

    def crop(self, bnd_box: Box, ext: int = 5) -> Any:
        """Crop the page using the provided bounding box and extension."""
        return box_ops.crop_img(self.page, bnd_box, ext=ext)

    def lcrop(self, indices: List[int], ext: int = 5) -> Any:
        """
        Returns the cropped image for the union of bounding boxes specified by the list of indices.
        
        Args:
            indices: List of indices for bounding boxes to be merged.
            ext: Extension padding to be applied after merging.
        """
        merged_box = self.union_boxes(indices)
        return self.crop(merged_box, ext=ext)

    @lru_cache(maxsize=32)
    def classify(self, index: int) -> str:
        """
        Classify the cropped image at the given index using OCR and return the result in lowercase.
        """
        return ocr.classify_image(self[index]).lower()

    @lru_cache(maxsize=32)
    def is_figure(self, index: int) -> bool:
        """
        Determines if the cropped image at the given index likely represents a figure.
        """
        if self.text_density(index)<0.8: 
            text = 'fig. text_density<0.8'
        else:
            text = self.classify(index)
        return text.startswith('fig.') or text.startswith('figure')

    @lru_cache(maxsize=32)
    def is_table(self, index: int) -> bool:
        """
        Determines if the cropped image at the given index likely represents a table.
        """
        text = self.classify(index)
        return text.startswith('table')

    @lru_cache(maxsize=32)
    def transcribe(self, index: int) -> str:
        """
        Transcribe the text from the cropped image at the given index using OCR.
        """
        if self.page_hr is None:
            return ocr.image_to_text(self[index])
        else:
            img = box_ops.crop_img(self.page_hr, np.array(self.bnd_boxes[index])*3, ext=15)
            #cv2.imshow('',img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            text = ocr.image_to_text(img)
            _ = text.replace('\n', '').replace(' ', '')
            total_area = img.shape[0] * img.shape[1]/9
            den2 = len(_) / total_area * 1E3
            print('Density =', den2)
            if den2>4.2:
                text = ocr.tesseract(img)
                print('** HIGH DENSITY TEXT **')
                print("Tesseract:", text)
            return text

    def union_boxes(self, indices: List[int], ext: int = 0) -> Box:
        """
        Compute the union of multiple bounding boxes specified by the given indices.
        
        Args:
            indices: List of bounding box indices.
            ext: Optional extension to apply after merging.
            
        Returns:
            The union of the bounding boxes extended by 'ext'.
        """
        merged_box = self.bnd_boxes[indices[0]]
        for i in indices[1:]:
            merged_box = box_ops.union_boxes(merged_box, self.bnd_boxes[i])
        merged_box = box_ops.ext_box(self.page, merged_box, ext=ext)
        return merged_box

    @lru_cache(maxsize=32)
    def is_small_block(self, index: int) -> bool:
        """
        Determines if the cropped image at the given index represents a small block,
        using both size constraints and OCR text length.
        """
        top, bottom, left, right = self.bnd_boxes[index]
        is_small_size = (right - left < self.width / 2 and bottom - top < 150)
        # Remove spaces and strip to get actual character count
        text_length = len(ocr.tesseract(self[index]).strip().replace(" ", ""))
        is_small_text = text_length < 50
        print("IS SMALL?", is_small_size, is_small_text, text_length, (right-left), bottom-top)
        return is_small_size and is_small_text


    @lru_cache(maxsize=32)
    def text_density(self, i):
        # Convert to grayscale
        image = self[i]
        _ = ocr.tesseract(image).replace('\n', '').replace(' ', '')
        total_area = image.shape[0] * image.shape[1]
        return len(_) / total_area * 1E3

    def clear_cache(self) -> None:
        """
        Clears the cache for all lru_cached methods.
        """
        self.classify.cache_clear()
        self.is_figure.cache_clear()
        self.is_table.cache_clear()
        self.transcribe.cache_clear()
        self.is_small_block.cache_clear()
        self.text_density.cache_clear()
