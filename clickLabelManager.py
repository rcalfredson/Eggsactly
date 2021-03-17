class ClickLabelManager:
    """Track clicks for annotating egg positions."""

    def __init__(self):
        """Create new ClickLabelManager instance."""
        self.clicks = dict()
        self.categories = ('Blurry', 'Difficult')
        for c in self.categories:
            setattr(self, 'is%sLabels' % c, dict())
        self.chamberTypes = dict()
        self.frontierData = {}

    @staticmethod
    def subImageKey(imageName, rowNum, colNum, position=None):
        """Return the key for a list of egg-position clicks for a single sub-image.

        Arguments:
          - imageName: Basename of the image
          - rowNum: Row number of the sub-image being annotated
          - colNum: Column number of the sub-image being annotated
          - position: Optional string to represent relative position of the egg-
                      laying area to the central well (e.g., "upper"). Used for
                      cases where multiple egg-laying areas share one set of row and
                      column indices.
        """
        key = '%s_%i_%i' % (imageName, rowNum, colNum)
        if position is not None and len(position) > 0:
            key += "_%s" % position
        return key

    def setChamberType(self, imageName, ct):
        """Assign chamber type for the given image.

        Arguments:
          - imageName: Basename of the image
          - ct: a chamber type (see chamber.py)
        """
        self.chamberTypes[imageName] = ct

    def addKey(self, imageName, rowNum, colNum, position=None):
        """Add an empty list of egg-position clicks for a single sub-image.

        Arguments:
          - imageName: Basename of the image
          - rowNum: Row number of the sub-image being annotated
          - colNum: Column number of the sub-image being annotated
          - position: Optional string to represent relative position of the egg-
                      laying area to the central well (e.g., "upper"). Used for
                      cases where multiple egg-laying areas share one set of row and
                      column indices.
        """
        key = self.subImageKey(imageName, rowNum, colNum)
        self.clicks[key] = []

    def addCategoryLabel(self, imageName, rowNum, colNum, category, isPositive,
                         position=None):
        key = self.subImageKey(imageName, rowNum, colNum, position)
        getattr(self, 'is%sLabels' % category)[key] = isPositive

    def addClick(self, imageName, rowNum, colNum, coords, position=None):
        """Associate a click with an egg on the given sub-image.

        Arguments:
          - imageName: Basename of the image
          - rowNum: Row number of the sub-image being annotated
          - colNum: Column number of the sub-image being annotated
        """
        key = self.subImageKey(imageName, rowNum, colNum, position)
        if key in self.clicks:
            self.clicks[key].append(coords)
        else:
            self.clicks[key] = [coords]

    def clearClicks(self, imageName, rowNum, colNum, position=None):
        """Delete the list of egg-position clicks for a single sub-image

        Arguments:
          - imageName: Basename of the image
          - rowNum: Row number of the sub-image being annotated
          - colNum: Column number of the sub-image being annotated
          - position: Optional string to represent relative position of the egg-
                      laying area to the central well (e.g., "upper"). Used for
                      cases where multiple egg-laying areas share one set of row and
                      column indices.
        """
        key = self.subImageKey(imageName, rowNum, colNum, position)
        if key not in self.clicks:
            return
        self.clicks.pop(self.subImageKey(imageName, rowNum, colNum, position))

    def clearClick(self, imageName, rowNum, colNum, index, position=None):
        """Delete a click for a single sub-image at the given index.

        Arguments:
          - imageName: Basename of the image
          - rowNum: Row number of the sub-image being annotated
          - colNum: Column number of the sub-image being annotated
          - index: List index of the click to remove
        """
        key = self.subImageKey(imageName, rowNum, colNum, position)
        if key not in self.clicks:
            return
        self.clicks[key].remove(self.clicks[key][index])

    def getClicks(self, imageName, rowNum, colNum, position=None):
        """Return the list of egg-position clicks for a single-sub-image.

        Arguments:
          - imageName: Basename of the image
          - rowNum: Row number of the sub-image being annotated
          - colNum: Column number of the sub-image being annotated
        """
        key = self.subImageKey(imageName, rowNum, colNum, position)
        if key not in self.clicks:
            self.clicks[key] = []
        return self.clicks[key]

    def getNumClicks(self, imageName, rowNum, colNum, position=None):
        """Return the number of egg-position clicks for a single sub-image (or None
        if the image has not been clicked yet).

        Arguments:
          - imageName: Basename of the image
          - rowNum: Row number of the sub-image being annotated
          - colNum: Column number of the sub-image being annotated
        """
        key = self.subImageKey(imageName, rowNum, colNum, position)
        if not key in self.clicks:
            return None
        else:
            return len(self.clicks[key])
