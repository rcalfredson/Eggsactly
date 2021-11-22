def progressBarDisplay(self):
    """Display green bar in the UI indicating proportion of annotated images."""
    proportionComplete = (self.absolutePosition() + 1) / self.totalNumSubImgs()
    cv2.rectangle(self.img, (0, self.img.shape[0] - 20),
      (self.img.shape[1] - 1, self.img.shape[0]), COL_G, 1)
    cv2.rectangle(self.img, (0, self.img.shape[0] - 20),
      (round(self.img.shape[1] * proportionComplete), self.img.shape[0]),
      COL_G, cv2.FILLED)

def absolutePosition(self):
    """Return ordinal position of the current sub-image across all images."""
    return sum([len(eggCountList) for eggCountList in self.eggCounts[
        :self.imgIdx]]) + self.subImgIdx