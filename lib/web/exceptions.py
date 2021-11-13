class CUDAMemoryException(Exception):
    pass


class ImageAnalysisException(Exception):
    pass


class ImageIgnoredException(Exception):
    pass


errorMessages = {
    ImageAnalysisException: "Image could not be analyzed.",
    CUDAMemoryException: "Error: system ran out of resources",
    ImageIgnoredException: "Image marked as ignored.",
}
