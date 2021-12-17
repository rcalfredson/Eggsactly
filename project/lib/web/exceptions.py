class CUDAMemoryException(Exception):
    pass


class ImageAnalysisException(Exception):
    pass


class ImageIgnoredException(Exception):
    pass


errorMessages = {
    CUDAMemoryException: "Error: system ran out of resources",
    ImageAnalysisException: "Image could not be analyzed.",
    ImageIgnoredException: "Image marked as ignored.",
}
