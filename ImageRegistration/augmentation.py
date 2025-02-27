import SimpleITK as sitk
import torchio as tio


def save_augmentation(aug_img: object, orig_path: str, extension: str) -> str:
    dot_idx = orig_path.rfind(".")
    new_path = orig_path[:dot_idx] + extension + orig_path[dot_idx:]
    sitk.WriteImage(aug_img, new_path)
    return new_path


def reldef(img_path: str, deform_params: dict = {"num_control_points": (7, 7, 7), "locked_borders": 2},
           ext: str = "_aug_reldef") -> str:
    img = sitk.ReadImage(img_path)
    deform = tio.RandomElasticDeformation(**deform_params)
    aug_img = deform(img)
    aug_path = save_augmentation(aug_img, img_path, extension=ext)
    return aug_path


def normalise(img_path: str, ext: str = "_aug_norm") -> str:
    img = sitk.ReadImage(img_path)
    norm_filter = sitk.NormalizeImageFilter()
    aug_img = norm_filter.Execute(img)
    aug_path = save_augmentation(aug_img, img_path, extension=ext)
    return aug_path


def rescale(img_path: str, bounds: tuple[int], ext: str = "_aug_rescale") -> str:
    img = sitk.ReadImage(img_path)
    aug_img = sitk.RescaleIntensity(img,bounds[0], bounds[1])
    aug_path = save_augmentation(aug_img, img_path, extension=ext)
    return aug_path


def random_gamma(img_path: str, bounds: tuple[float], ext: str = "_aug_rgam") -> str:
    img = sitk.ReadImage(img_path)
    rgam_transform = tio.transforms.RandomGamma(log_gamma=bounds)
    aug_img = rgam_transform(img)
    aug_path = save_augmentation(aug_img, img_path, extension=ext)
    return aug_path


def random_rotation(img_path: str, degree_bounds: tuple[int], ext: str = "_aug_rot") -> str:
    img = sitk.ReadImage(img_path)
    rotation = tio.RandomAffine(degrees=degree_bounds, scales=0)
    aug_img = rotation(img)
    aug_path = save_augmentation(aug_img, img_path, extension=ext)
    return aug_path
