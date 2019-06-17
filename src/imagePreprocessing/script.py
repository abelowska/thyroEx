import os
from collections import Counter

from src.imagePreprocessing.utils import rotate, crop_image_after_rotation, resize_image, save_image, read_image
from src.imagePreprocessing.tickBarsDetector import ImageResizerFactory
from src.imagePreprocessing.annotationsRemover import AnnotationRemoverCreator


PATH = ''
ANGLE = 10
SCALE = 224


files_list = os.listdir(PATH)  # returns list
image_resizer = ImageResizerFactory().columbia_images()
ticks = []

# finding scale
for file in files_list:
    image = read_image(PATH + file)
    tick = image_resizer.find_tick(image)
    image_resizer.save_bar_tick(PATH + file, tick)
    ticks.append(tick)

# find most common scale
diffs_dict = Counter(ticks)
default_bar_tick = diffs_dict.most_common(1)[0][0]


# removing artifacts and normalizing
annotation_remover = AnnotationRemoverCreator.columbia_annotations()

for file in files_list:
    image = read_image(PATH + file)
    image = annotation_remover.remove_annotations(image)

    metadata_tick = image_resizer.read_tick(PATH + file)
    image = image_resizer.resize(metadata_tick, default_bar_tick, image)
    save_image(image, PATH + file)

# augmentation

for file in files_list:
    image = read_image(PATH + file)
    images = rotate(image, ANGLE)
    i = 0
    for im in images:
        im = crop_image_after_rotation(im)
        filename, file_extension = os.path.splitext(PATH + file)
        save_image(im, '{}{}_{}{}'.format(PATH, filename, i, file_extension))
        i += 1

# downscaling
files_list = os.listdir(PATH)  # returns list

for file in files_list:
    image = read_image(PATH + file)
    image = resize_image(image, 224)
    save_image(image, PATH + file)
