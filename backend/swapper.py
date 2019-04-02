from image_engine import BingImageEngine

class Swapper:
    def __init__(self):
        self.src_img = None
        key = "f956df8f4aa74af396fbc1a5072e7960"
        self.engine = BingImageEngine(key)

    def request_images(self, item):
        self.engine.request(item)

    def swap(self, idx):
        swapped = self.src_img
        tgt_img = self.engine.iterate(idx)
        # invoke faceswapping (self.src_img, tgt_img) 
        return swapped

    def compare(self, idx):
        tgt_img = self.engine.iterate(idx)
        # invoke faceswapping (self.src_img, tgt_img) 
        return True
    
    def process_one(self, idx):
        output = None
        if self.compare(idx):
            output = self.swap(idx)
        return output

    def process(self, img, item):
        self.src_img = img
        if not self.request_images(item):
            print("Cannot search for images!")

        for idx in range(self.engine.length()):
            self.process_one(idx)



