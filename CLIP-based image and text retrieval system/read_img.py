from PIL import Image
import os

class ImageReader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def read_images(self):
        image_list = []

        for filename in os.listdir(self.folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(self.folder_path, filename)
                image = Image.open(file_path)
                image_list.append(image)
        return image_list