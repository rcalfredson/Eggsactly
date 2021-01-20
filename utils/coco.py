from .color import randColor

def getNewCategory(catId):
    return {'id': catId,
            'name': str(catId),
            'supercategory': '',
            'color': randColor(),
            'metadata': {},
            'keypoint_colors': []}