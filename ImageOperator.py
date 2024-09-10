from PIL import Image


class ImageOperator:
    def __init__(self, src_path, dst_folder_path, read, delete, new_name=None):
        self.read = read
        self.src_path = src_path
        self.name = new_name
        if not self.name:
            self.name = self.src_path.name
        self.dst_path = dst_folder_path / self.name
        self.image = None
        self.delete = delete

    def __enter__(self):
        if self.read:
            self.image = Image.open(self.src_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.read:
            self.image.save(self.dst_path)
            self.image.close()
            if self.delete:
                self.src_path.unlink()