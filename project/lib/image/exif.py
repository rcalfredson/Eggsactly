from PIL import Image, ExifTags

def correct_via_exif(filepath, read_only=False):
    try:
        image=Image.open(filepath)

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        
        exif = image._getexif()

        if exif[orientation] == 3:
            if not read_only:
                image=image.rotate(180, expand=True)
            else:
                print('rotating by 180')
        elif exif[orientation] == 6:
            if not read_only:
                image=image.rotate(270, expand=True)
            else:
                print('rotating by 270')
        elif exif[orientation] == 8:
            if not read_only:
                image=image.rotate(90, expand=True)
            else:
                print('rotating by 90')
        if not read_only:
            image.save(filepath)
        image.close()
    except (TypeError, AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass
