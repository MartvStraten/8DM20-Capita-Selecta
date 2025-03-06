import SimpleITK as sitk
import torchio as tio


def save_augmentation(aug_img: object, orig_path: str, extension: str) -> str:
    '''Saves (augmented) image by inserting extension into original path.
    
    parameters
    aug_img: image object
    orig_path: file path of unaugmented image
    extension: text to be added after original filepath when saving new image
    
    returns
    new_path: file path of the saved augmented image
    '''
    dot_idx = orig_path.rfind(".")
    new_path = orig_path[:dot_idx] + extension + orig_path[dot_idx:]
    sitk.WriteImage(aug_img, new_path)
    return new_path


def reldef(img_path: str, mask_path: str, deform_params: dict = {"num_control_points": 7, "max_displacement": 10, 
            "locked_borders": 2}, ext: str = "_aug_reldef") -> str:
    '''Random elastic deformation of image and mask located at input file path.

    parameters
    img_path: file path to image
    mask_path: file path to mask
    deform_params: dictionary of parameters for 'tio.RandomElasticDeformation()'
    ext: text to be added after original filepath when saving new image
    
    returns
    aug_img_path: file path of the saved augmented image
    aug_mask_path: file path of the saved augmented mask
    '''
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)
    deform = tio.RandomElasticDeformation(**deform_params)
    aug_img = deform(img)
    aug_mask = deform(mask)
    aug_img_path = save_augmentation(aug_img, img_path, extension=ext)
    aug_mask_path = save_augmentation(aug_mask, mask_path, extension=ext)
    return aug_img_path, aug_mask_path


def normalise(img_path: str, ext: str = "_aug_norm") -> str:
    '''Normalise (mean=0, var=1) image located at input file path.

    parameters
    img_path: file path to image
    ext: text to be added after original filepath when saving new image
    
    returns
    aug_path: file path of the saved augmented image
    '''
    img = sitk.ReadImage(img_path)
    norm_filter = sitk.NormalizeImageFilter()
    aug_img = norm_filter.Execute(img)
    aug_path = save_augmentation(aug_img, img_path, extension=ext)
    return aug_path


def rescale(img_path: str, bounds: tuple[int], ext: str = "_aug_rescale") -> str:
    '''Rescale pixel values of image located at input file path.

    parameters
    img_path: file path to image
    bounds: tuple of new min and max pixel intensity
    ext: text to be added after original filepath when saving new image
    
    returns
    aug_path: file path of the saved augmented image
    '''
    img = sitk.ReadImage(img_path)
    aug_img = sitk.RescaleIntensity(img,bounds[0], bounds[1])
    aug_path = save_augmentation(aug_img, img_path, extension=ext)
    return aug_path


def random_gamma(img_path: str, bounds: tuple[float], ext: str = "_aug_rgam") -> str:
    '''Appply random gamma correction to image located at input file path.

    parameters
    img_path: file path to image
    bounds: tuple of min and max gamma
    ext: text to be added after original filepath when saving new image
    
    returns
    aug_path: file path of the saved augmented image
    '''
    img = sitk.ReadImage(img_path)
    rgam_transform = tio.transforms.RandomGamma(log_gamma=bounds)
    aug_img = rgam_transform(img)
    aug_path = save_augmentation(aug_img, img_path, extension=ext)
    return aug_path


def random_rotation(img_path: str, bounds: tuple[int], ext: str = "_aug_rot") -> str:
    '''Appply random rotation to image located at input file path.

    parameters
    img_path: file path to image
    bounds: tuple of min and max rotation in every axis
    ext: text to be added after original filepath when saving new image
    
    returns
    aug_path: file path of the saved augmented image
    '''
    img = sitk.ReadImage(img_path)
    rotation = tio.RandomAffine(degrees=bounds, scales=0)
    aug_img = rotation(img)
    aug_path = save_augmentation(aug_img, img_path, extension=ext)
    return aug_path
