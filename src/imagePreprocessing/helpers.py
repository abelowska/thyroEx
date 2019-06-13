import cv2


# crop to square without black padding after rotation
def crop_image(image):
    y_size, x_size, _ = image.shape
    dest_size = min(y_size, x_size)

    # TODO automatically calculated add margin
    margin_x = x_size - dest_size + 70
    margin_y = y_size - dest_size + 70

    print('{}, {}'.format(margin_x, margin_y))

    return image[int(margin_y / 2):y_size - int(margin_y / 2), int(margin_x / 2):x_size - int(margin_x / 2)]


def resize_image(image, size):
    return cv2.resize(image, (size, size))


def rotation_augmentation(image, max_angle):
    y_size, x_size, _ = image.shape
    angle_list = range(-max_angle, max_angle)

    print(angle_list)

    matrix_list = [cv2.getRotationMatrix2D((x_size / 2, y_size / 2), angle, 1) for angle in angle_list]
    images = [cv2.warpAffine(image.copy(), M, (x_size, y_size)) for M in matrix_list]

    return images
