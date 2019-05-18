from image_engine import BingImageEngine
import face_swap_py as fspy

landmarks_path = 'data/shape_predictor_68_face_landmarks.dat'
model_3dmm_h5_path = 'data/BaselFaceModel_mod_wForehead_noEars.h5'
model_3dmm_dat_path = 'data/BaselFace.dat'
reg_model_path = 'data/3dmm_cnn_resnet_101.caffemodel'
reg_deploy_path = 'data/3dmm_cnn_resnet_101_deploy.prototxt'
reg_mean_path = 'data/3dmm_cnn_resnet_101_mean.binaryproto'
seg_model_path = 'data/face_seg_fcn8s_300.caffemodel'          # or 'data/face_seg_fcn8s_300.caffemodel' for lower resolution
seg_deploy_path = 'data/face_seg_fcn8s_300_deploy.prototxt'    # or 'data/face_seg_fcn8s_300_deploy.prototxt' for lower resolution
generic = False
with_expr = False
with_gpu = True
gpu_device_id = 0


class Swapper:
    def __init__(self):
        self.src_img = None
        self.cur_tgt = None
        self.fs_engine = None
        self.img_engine = None

    def start_fs_engine(self):
        self.fs_engine = fspy.FaceSwap(landmarks_path, model_3dmm_h5_path,
                        model_3dmm_dat_path, reg_model_path,
                        reg_deploy_path, reg_mean_path,
                        seg_model_path, seg_deploy_path,
                        generic, with_expr, with_gpu, gpu_device_id)
    
    def start_img_engine(self):
        key = "3e49862cde1443c3b88b0538bb42b6cd"
        self.img_engine = BingImageEngine(key)
    
    def set_image(self, img):
        img = fspy.FaceData(img)
        self.fs_engine.estimate(img)
        return img

    def request_images(self, item):
        return self.img_engine.request(item)

    def swap(self, src_img, cur_tgt):
        swapped = src_img
        # invoke faceswapping (self.src_img, tgt_img) 
        swapped = self.fs_engine.transfer(src_img, cur_tgt)
        print("swapped")
        print(swapped.shape)
        print(type(swapped))
        return swapped

    def compare(self, src_img, tgt_img):
      #  cur_tgt = fspy.FaceData(tgt_img)
      #  self.fs_engine.estimate(cur_tgt)
        # invoke faceswapping (self.src_img, tgt_img) 
        ret = self.fs_engine.compare(src_img, tgt_img)
        #return True
        return ret

    def process_one(self, src_img, tgt_img):
        output = None
        if self.compare(src_img, tgt_img):
            output = self.swap(src_img, tgt_img)
        return output

    def process(self, img, item):
        src_img = self.set_image(img)
        tgt_imgs = self.request_images(item)
        if not tgt_imgs:
            print("Cannot search for images!")

        for idx in range(len(tgt_imgs)):
            self.process_one(src_img, tgt_imgs[idx])



if __name__ == "__main__":

    fs_engine = fspy.FaceSwap(landmarks_path, model_3dmm_h5_path,
                    model_3dmm_dat_path, reg_model_path,
                    reg_deploy_path, reg_mean_path,
                    seg_model_path, seg_deploy_path,
                    generic, with_expr, with_gpu, gpu_device_id)
    
    import cv2
    img1 = fspy.FaceData(cv2.imread('trump1.jpg'))
    img2 = fspy.FaceData(cv2.imread('xi1.jpg'))
    fs_engine.estimate(img1)
    fs_engine.estimate(img2)


