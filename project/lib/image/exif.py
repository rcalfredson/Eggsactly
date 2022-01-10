from io import BytesIO
from PIL import ExifTags, Image


def correct_via_exif(data=None, path=None, format=None, read_only=False):
    """rotate an image by the amount specified in its EXIF data.

    Arguments:
      data: binary image data
      path: path to an image file
      format: format to use for the rotated image. Defaults to the format
              determined by PIL.
      read_only: print the EXIF rotation data (if found) without
                 modifying the image.

    note: `data` and `path` are mutually exclusive arguments, and
    it's required to specify one of them.
    """
    if data is None and path is None:
        raise TypeError("must specify either data or path")
    if data is not None and path is not None:
        raise ValueError("either data or path must be specified, not both.")
    image = Image.open(BytesIO(data) if data is not None else path)
    if type(format) is str and format.lower() == "jpg":
        format = "jpeg"
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        exif = image.getexif()

        if exif[orientation] == 3:
            if not read_only:
                image = image.rotate(180, expand=True)
            else:
                print("rotating by 180")
        elif exif[orientation] == 6:
            if not read_only:
                image = image.rotate(270, expand=True)
            else:
                print("rotating by 270")
        elif exif[orientation] == 8:
            if not read_only:
                image = image.rotate(90, expand=True)
            else:
                print("rotating by 90")
        if not read_only:
            kwargs = {"format": format} if format is not None else {}
            if path is not None:
                image.save(path, **kwargs)
                image.close()
            else:
                output = BytesIO()
                image.save(output, **kwargs)
                image.close()
                return output.getvalue()
    except (TypeError, AttributeError, KeyError, IndexError):
        # image doesn't have EXIF
        if data is not None:
            return data
