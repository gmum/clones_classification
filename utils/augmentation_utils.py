import numpy.random as random


class RandomRotate(object):
    """Rotate the given PIL.Image by either 0, 90, 180, 270."""

    def __call__(self, img):
        random_rotation = random.randint(4, size=1)
        if random_rotation == 0:
            pass
        else:
            img = img.rotate(random_rotation*90)
        return img

