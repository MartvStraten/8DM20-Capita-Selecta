import SimpleITK as sitk
import torchio as tio

def save_augmentation(aug_img, orig_path, extension):
    new_path = orig_path[:-4] + extension + orig_path[-4:]
    print(new_path)
    sitk.WriteImage(aug_img, new_path)
    return new_path

def reldef(img_path):
    img = sitk.ReadImage(img_path)
    deform = tio.RandomElasticDeformation(num_control_points=(7, 7, 7), locked_borders=2)
    aug_img = deform(img)
    aug_path = save_augmentation(aug_img, img_path, extension="_aug_reldef")
    return aug_path
