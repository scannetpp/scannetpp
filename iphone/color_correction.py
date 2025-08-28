# TODO: Fix the problem when the image has mask (avoid using the masked regions for color correction)
# TODO: Make it a independent script for evaluation
# TODO: Remove the hardcoded paths and numbers

from typing import List, Optional, Union
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import ot

from common.scene_release import ScannetppScene_Release
from eval.nvs import get_test_images


SUPPORT_IMAGE_FORMAT = [".JPG", ".jpg", ".png", ".PNG", ".jpeg"]
ALL_PIXELS = 2764800  # in case of (1440, 1920)


def get_concat_h(im1, im2, im3=None):
    """
    Concatenate images horizontally.

    Args:
        im1: iPhone image
        im2: Undistorted DSLR by im1
        im3: Color corrected im1 (optional)

    Returns:
        PIL.Image: Concatenated image
    """
    if im3 is not None:
        dst = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        dst.paste(im3, (im1.width + im2.width, 0))
        return dst
    else:
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst


class COLOR_CORRECTOR:
    def __init__(
        self,
        method="default",
        option=None,
        sample_size=500,
        batch_size=10,
        dummy_dslr_path="",
        mode="",
    ):
        """
        Initialize color corrector.

        Args:
            method: Color correction method. Options:
                "default": POT (Python Optimal Transport)
                    - requires option in [None, EmdTransport, SinkhornTransport,
                      MappingTransport_linear, MappingTransport_gaussian]
                "linear": linear system
            option: Specific transport method option
            sample_size: Number of pixels to sample for color correction
            batch_size: Processing batch size
            dummy_dslr_path: Path to dummy DSLR image
            mode: Processing mode
        """
        self.system = None
        self.method = method
        self.option = option
        self.batch_size = batch_size
        self.mode = mode
        self.dummy_dslr_path = dummy_dslr_path
        self.sample_size = sample_size
        self.all_sample_size = None
        self.im_shape = None

    def __call__(self, Xs, Xt):
        """
        Fit the color correction system.

        Args:
            Xs: Source images array with shape [:, 3]
            Xt: Target images array with shape [:, 3]
        """
        print(self.method, self.option)
        if self.method == "default":
            assert self.option is not None
            if self.option == "EmdTransport":
                self.system = ot.da.EMDTransport()
                self.system.fit(Xs=Xs, Xt=Xt)
            elif self.option == "SinkhornTransport":
                self.system = ot.da.SinkhornTransport(reg_e=1e-1)
                self.system.fit(Xs=Xs, Xt=Xt)
            elif self.option == "MappingTransport_linear":
                # max_iter = 20
                self.system = ot.da.MappingTransport(
                    mu=1e0, eta=1e-8, bias=True, max_iter=10, verbose=True)
                self.system.fit(Xs=Xs, Xt=Xt)
            elif self.option == "MappingTransport_gaussian":
                # max_iter = 10
                self.system = ot.da.MappingTransport(
                    mu=1e0, eta=1e-2, sigma=1, bias=False, max_iter=5, verbose=True)
                self.system.fit(Xs=Xs, Xt=Xt)
            elif self.option == "LinearGWTransport":
                self.system = ot.da.LinearGWTransport()
                self.system.fit(Xs=Xs, Xt=Xt)
            elif self.option == "LinearTransport":
                self.system = ot.da.LinearTransport()
                self.system.fit(Xs=Xs, Xt=Xt)
        # elif self.method == "linear":
        #     assert self.option is not None
        #     if self.option == "SVD":
        #         self.system = Procrustes()
        #         self.system.fit(Xs=Xs, Xt=Xt)
        # elif self.method == "histogram":
        #     assert self.option == "scikit"
        #     self.system = Histogram()

    def transform_and_result(self, eval_Xs, eval_Xt, eval_masks, eval_batch_size=100):
        """
        Apply color correction transformation.

        Args:
            eval_Xs: Evaluation source images
            eval_Xt: Evaluation target images
            eval_masks: Evaluation masks
            eval_batch_size: Batch size for evaluation

        Returns:
            List of transformed images
        """
        print(len(eval_Xs))

        # iPhone (source to be color-corrected)
        Xs = np.array(eval_Xs).reshape(eval_batch_size, self.all_sample_size, 3)
        # DSLR (source to be used for estimating cc-function) - not used in current implementation
        # Xt = np.array(eval_Xt).reshape(eval_batch_size, self.all_sample_size, 3)
        # iPhone, magenta == 1.0
        eval_masks = np.array(eval_masks)

        transformed_Xs = []
        for i in range(eval_batch_size):
            _out = None
            _mask = eval_masks[i]  # already image shape

            # if self.method == "histogram":
            #     image = self.mat2im(Xs[i], self.im_shape)
            #     reference = self.mat2im(Xt[i], self.im_shape)
            #     out = match_histograms(image, reference, channel_axis=-1)
            #     print("out red: ", out[:, :, 0].min(), out[:, :, 0].max())
            #     print("out blue: ", out[:, :, 1].min(), out[:, :, 1].max())
            #     print("out green: ", out[:, :, 2].min(), out[:, :, 2].max())
            #     _out = self.minmax(out)
            # else:
            X1 = Xs[i]  # [H*W, 3]
            _out = self.minmax(self.mat2im(self.system.transform(Xs=X1), self.im_shape))

            print("out: ", _out.shape)
            _out[_mask] = [1.0, 0.5, 1.0]  # magenta
            transformed_Xs.append((_out * 255).astype(np.uint8))
        return transformed_Xs

    def preparation(self, path2trainIs=None, path2trainIt=None, path2trainmasks=None,
                    path2evalIs=None, path2evalIt=None, path2evalmasks=None,
                    sample_every=4, paired=True, offset=0, mode="",
                    train_batch_size=10, eval_batch_size=10, downsample=-1):
        """
        Prepare training and evaluation data for color correction.

        Args:
            path2trainIs: List of training iPhone image paths
            path2trainIt: List of training DSLR image paths
            path2trainmasks: List of training mask paths
            path2evalIs: List of evaluation iPhone image paths
            path2evalIt: List of evaluation DSLR image paths
            path2evalmasks: List of evaluation mask paths
            sample_every: Sample every N pixels
            paired: Whether to use paired pixel selection
            offset: Starting offset for processing
            mode: Processing mode
            train_batch_size: Training batch size
            eval_batch_size: Evaluation batch size
            downsample: Downsampling factor (-1 for no downsampling)

        Returns:
            Tuple of prepared data arrays and paths
        """
        # Initialize defaults for mutable arguments
        if path2trainIs is None:
            path2trainIs = []
        if path2trainIt is None:
            path2trainIt = []
        if path2trainmasks is None:
            path2trainmasks = []
        if path2evalIs is None:
            path2evalIs = []
        if path2evalIt is None:
            path2evalIt = []
        if path2evalmasks is None:
            path2evalmasks = []

        Xs = None
        Xt = None

        print("path2trainIs: ", len(path2trainIs))
        print("path2trainIt: ", len(path2trainIt))
        print("sample_size: ", self.sample_size, ALL_PIXELS // sample_every)
        all_samples = len(path2trainIs)
        print("all samples: ", all_samples)

        train_pairs_save_dir = os.path.join(os.path.dirname(path2trainIs[0]), "train_pairs")
        if not os.path.exists(train_pairs_save_dir):
            os.makedirs(train_pairs_save_dir)

        for tn in range(offset, offset + train_batch_size):
            """
            Process training data:
            - Xs: train source images *applied mask (#sample_size*train_batch_size, 3)
            - Xt: train target images *applied mask (#sample_size*train_batch_size, 3)
            """
            print("---train---")
            path2is = path2trainIs[tn]  # iPhone
            if str(path2is)[-10] == "_":
                path2is = Path(str(path2is)[:-9] + "0" + str(path2is)[-9:])

            path2it = path2trainIt[tn]  # DSLR
            path2mask = path2trainmasks[tn]  # common mask
            has_diff_imsize = False

            I1 = plt.imread(path2is).astype(np.float64) / 255  # iPhone
            I2 = plt.imread(path2it).astype(np.float64) / 255  # DSLR
            save_im_path = os.path.join(
                train_pairs_save_dir,
                f"{Path(path2is).stem}_{Path(path2it).stem}.png")
            w = I2.shape[1]
            h = I2.shape[0]
            print("I1 shape: ", I1.shape)
            print("I2 shape: ", I2.shape)
            if I1.shape != I2.shape:
                has_diff_imsize = True
                I1 = np.array(Image.open(path2is).convert("RGB").resize(
                    (I2.size[1], I2.size[0]))).astype(np.float64)
                I1 = I1 / 255
                print("I1 shape: ", I1.shape)
                print("I2 shape: ", I2.shape)

            if downsample > 0:
                if not has_diff_imsize:
                    I1 = Image.open(path2is)

                I1 = np.array(I1.resize((w // downsample, h // downsample))).astype(np.float64)
                I1 = I1 / 255
                downsampled_w = w // downsample
                downsampled_h = h // downsample
                I2 = Image.open(path2it)
                I2 = np.array(I2.resize((w // downsample, h // downsample))).astype(np.float64)
                I2 = I2 / 255

                print("I1 downsampled: ", I1.shape)
                print("I2 downsampled: ", I2.shape)

            concat_train_pair = get_concat_h(
                im1=Image.fromarray(I1.astype(np.uint8)).convert("RGB"),
                im2=Image.fromarray(I2.astype(np.uint8)).convert("RGB"),
                im3=None
            )
            concat_train_pair.save(save_im_path)

            _Xs = self.im2mat(I1)
            _Xt = self.im2mat(I2)

            if os.path.exists(path2mask):
                mask = plt.imread(path2mask).astype(bool)  # [H, W, 1]
                if downsample > 0:
                    mask = np.array(Image.open(path2mask).resize(
                        (downsampled_w, downsampled_h))).astype(bool)
                print("downsampled mask: ", mask.shape)
            else:
                mask = np.ones_like(I1[..., 0]).astype(bool)

            _mask = mask.reshape(-1)

            print("source fname: ", path2is)
            print("target fname: ", path2it)
            print("source/target mask: ", path2mask)
            print("source, target, mask ", I1.shape, I2.shape, mask.shape)  # source, target, mask

            # Remove masking part from _Xs, _Xt (do not use masking pixels for estimation)
            _Xs = _Xs[_mask]
            _Xt = _Xt[_mask]

            # Sample every N pixels
            idx = None
            idx1 = None
            idx2 = None

            if paired:
                print("paired pixel selections")

                if _Xs.shape[0] <= _Xt.shape[0]:
                    idx = np.arange(0, _Xs.shape[0], sample_every)
                else:
                    idx = np.arange(0, _Xt.shape[0], sample_every)

                if tn == 0 or mode == "server":
                    Xs = np.array(_Xs[idx, :]).astype(np.float64).reshape(-1, 3)
                else:
                    Xs = np.concatenate((Xs, np.array(_Xs[idx, :]).astype(np.float64).reshape(-1, 3)), axis=0)

                if tn == 0 or mode == "server":
                    Xt = np.array(_Xt[idx, :]).astype(np.float64).reshape(-1, 3)
                else:
                    Xt = np.concatenate((Xt, np.array(_Xt[idx, :]).astype(np.float64).reshape(-1, 3)), axis=0)
                print("actual sampled pixels: ", len(idx))
            else:
                print("unpaired random pixel selections")

                # Note: rng is not defined in this scope - this would need to be fixed
                # idx1 = rng.randint(_Xs.shape[0], size=(self.sample_size,))
                # idx2 = rng.randint(_Xt.shape[0], size=(self.sample_size,))
                rng = np.random.default_rng()
                idx1 = rng.integers(0, _Xs.shape[0], size=self.sample_size)
                idx2 = rng.integers(0, _Xt.shape[0], size=self.sample_size)

                if tn == 0 or mode == "server":
                    Xs = np.array(_Xs[idx1, :]).astype(np.float64).reshape(-1, 3)
                else:
                    Xs = np.concatenate((Xs, np.array(_Xs[idx1, :]).astype(np.float64).reshape(-1, 3)), axis=0)

                if tn == 0 or mode == "server":
                    Xt = np.array(_Xt[idx2, :]).astype(np.float64).reshape(-1, 3)
                else:
                    Xt = np.concatenate((Xt, np.array(_Xt[idx2, :]).astype(np.float64).reshape(-1, 3)), axis=0)
                print("actual sampled pixels: ", len(idx1), len(idx2))

            print("Xs: ", _Xs.shape)
            print("Xt: ", _Xt.shape)
            print("sample_size: ", self.sample_size)

            print("concat Xs: ", Xs.shape)
            print("concat Xt: ", Xt.shape)

        eval_Xs = []
        eval_Xt = []
        eval_masks = []

        eval_path2Is = []
        eval_path2It = []

        for en in range(offset, offset + eval_batch_size):  # eval_batch_size
            """
            Process evaluation data:
            - eval_Xs: eval source images (#sample_size*eval_batch_size, 3)
            - eval_Xt: train target images (#sample_size*eval_batch_size, 3)
            - eval_masks: inverted eval masks (H, W), magenta=True, else False
            """
            print("---eval---")
            print("index: ", en)

            path2is = path2evalIs[en]  # source
            path2it = path2evalIt[en]  # target
            path2mask = path2evalmasks[en]

            I1 = plt.imread(path2is).astype(np.float64) / 255  # iPhone
            I2 = plt.imread(path2it).astype(np.float64) / 255  # DSLR

            if I1.shape != I2.shape:
                print("I1 shape: ", I1.shape)
                print("I2 shape: ", I2.shape)
                I1 = np.array(Image.open(path2is).resize((I2.shape[1], I2.shape[0]))).astype(np.uint8)
                print("I1 shape: ", I1.shape)
                print("I2 shape: ", I2.shape)

            if self.im_shape is None:
                self.im_shape = I1.shape
                print(I1.shape)

            _Xs = self.im2mat(I1)
            _Xt = self.im2mat(I2)

            if self.all_sample_size is None:
                self.all_sample_size = _Xs.shape[0]
                print("all sample size: ", self.all_sample_size)

            if os.path.exists(path2mask):
                mask = plt.imread(path2mask).astype(bool)  # [H, W, 1]
            else:
                mask = np.ones_like(I1[..., 0]).astype(bool)

            inv_mask = np.logical_not(mask)
            eval_masks.append(inv_mask)  # iPhone, magenta == 1.0

            print("mask: ", inv_mask.shape)

            eval_Xs.append(_Xs)
            eval_Xt.append(_Xt)

            eval_path2Is.append(path2is)
            eval_path2It.append(path2it)

        Xs = np.array(Xs).astype(np.float64).reshape(-1, 3)
        Xt = np.array(Xt).astype(np.float64).reshape(-1, 3)

        eval_Xs = np.array(eval_Xs).astype(np.float64).reshape(-1, 3)
        eval_Xt = np.array(eval_Xt).astype(np.float64).reshape(-1, 3)
        eval_masks = np.array(eval_masks).astype(bool)

        print(f"Take {self.sample_size} pixels as data for estimation")
        print("train, train_Xs, train_Xt: ", train_batch_size, Xs.shape[0], Xt.shape[0])
        assert Xs.shape[0] == Xt.shape[0]

        print("eval, eval_Xs, eval_Xt: ", eval_batch_size, eval_Xs.shape[0], eval_Xt.shape[0])
        assert eval_Xs.shape[0] == eval_Xt.shape[0]

        print("eval, mask: ", eval_batch_size, eval_masks.shape[0])
        assert eval_batch_size == eval_masks.shape[0]

        return (Xs, Xt, eval_Xs, eval_Xt, eval_masks, eval_path2Is, eval_path2It,
                train_pairs_save_dir)  # [self.batch_size, -1, 3]

    @staticmethod
    def im2mat(img):
        """Converts an image to matrix (one pixel per line)."""
        return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

    @staticmethod
    def mat2im(X, shape):
        """Converts back a matrix to an image."""
        return X.reshape(shape)

    @staticmethod
    def minmax(img):
        """Clip image values to [0, 1] range."""
        return np.clip(img, 0, 1)


def color_correction(
    pred_dir: Union[str, Path],
    gt_dir: Union[str, Path],
    image_list: List[str],
    upload_path: Union[str, Path],
    mask_dir: Optional[Union[str, Path]] = None,
    scene_id: str = "unknown",
    verbose: bool = False,
    gt_file_format: str = ".JPG",
    device: str = "cpu",
    cc_configs: dict = None,
):
    """
    Apply color correction to predicted images.

    Args:
        pred_dir: Path to the directory containing the predicted images.
        gt_dir: Path to the directory containing the GT images.
        image_list: List of image names to evaluate.
        upload_path: Path to upload directory.
        mask_dir: Path to the directory containing the masks. Evaluate without mask if None.
        scene_id: Scene ID for logging.
        verbose: Print the evaluation results.
        gt_file_format: File format of the GT images.
        device: Device to use for computation.
        cc_configs: Color correction configuration dictionary.

    Process:
        1. Load pair of image/target
        2. Estimate optimal transport operator
        3. Apply operator on image/Save it in path2images_after_cc
        4. Create side-by-side dir/Save it
        5. Add the path2images_after_cc

    Returns:
        dict: Dictionary containing paths to original, GT, and color-corrected images.
    """
    if cc_configs is None:
        cc_configs = {
            "method": "default",
            "option": "LinearTransport",
            "sample_size": 2764800,
            "batch_size": 1,
            "mode": "server"
        }

    pred_after_cc_dir = Path(upload_path) / "color_corrected_pred"
    if not os.path.exists(pred_after_cc_dir):
        os.makedirs(pred_after_cc_dir)

    pred_after_cc_dir = Path(pred_after_cc_dir) / scene_id
    if not os.path.exists(pred_after_cc_dir):
        os.makedirs(pred_after_cc_dir)

    # Path to pairs (before/after)
    collection_dir = Path(upload_path) / "color_corrected_BA"
    if not os.path.exists(collection_dir):
        os.makedirs(collection_dir)

    collection_dir = Path(collection_dir) / scene_id
    if not os.path.exists(collection_dir):
        os.makedirs(collection_dir)

    # Copy color correction config
    with open(os.path.join(collection_dir, "cc_config.txt"), "w") as config_f:
        config_f.write(str(cc_configs))

    color_corrector = None
    path_dict = {"pred": [], "gt": [], "cc_pred": []}

    for _id, image_fn in enumerate(image_list):
        image_name = image_fn.split(".")[0]
        gt_image_path = os.path.join(gt_dir, image_name + gt_file_format)
        assert os.path.exists(
            gt_image_path
        ), f"{scene_id} GT image not found: {image_fn} given path {gt_image_path}"
        gt_image = Image.open(gt_image_path)

        pred_image_path = None
        for img_format in SUPPORT_IMAGE_FORMAT:
            test_image_path = os.path.join(pred_dir, image_name + img_format)
            if os.path.exists(test_image_path):
                pred_image_path = test_image_path
                break
        assert (
            pred_image_path is not None
        ), f"{scene_id} pred image not found: {image_fn} with the following format {' '.join(SUPPORT_IMAGE_FORMAT)}"
        pred_image = Image.open(pred_image_path)

        if mask_dir is not None:
            mask_path = os.path.join(mask_dir, image_name + ".png")
            assert os.path.exists(mask_path), f"mask not found: {mask_path}"
            mask = Image.open(mask_path)
            mask = torch.from_numpy(np.array(mask)).to(device)
            mask = (mask > 0).bool()
            assert (
                len(mask.shape) == 2
            ), f"mask should have 2 channels (H, W) but get shape: {mask.shape}"
            assert (
                mask.shape[0] == gt_image.size[1] and mask.shape[1] == gt_image.size[0]
            ), f"mask shape {mask.shape} does not match GT image size: {gt_image.size}"
        else:
            mask = None

        if gt_image.size != pred_image.size:
            # Auto resize to match the GT image size
            pred_image = pred_image.resize(gt_image.size, Image.BICUBIC)

        gt_image = torch.from_numpy(np.array(gt_image)).float() / 255.0
        gt_image = gt_image.to(device)
        pred_image = torch.from_numpy(np.array(pred_image)).float() / 255.0
        pred_image = pred_image.to(device)
        assert (
            len(gt_image.shape) == 3
        ), f"GT image should have 3 channels (H, W, 3) but get shape: {gt_image.shape}"
        assert (
            len(pred_image.shape) == 3
        ), f"pred image should have 3 channels (H, W, 3) but get shape: {pred_image.shape}"

        gt_image = gt_image.permute(2, 0, 1).unsqueeze(0)
        pred_image = pred_image.permute(2, 0, 1).unsqueeze(0)

        # If the mask is given and not all pixels are valid
        if mask is not None and not torch.all(mask):
            mask = mask.unsqueeze(0)  # (1, H, W)
            print("mask: ", mask.shape)
            valid_gt = torch.masked_select(gt_image, mask).view(3, -1).numpy()
            valid_pred = torch.masked_select(pred_image, mask).view(3, -1).numpy()
        else:
            valid_gt = gt_image.view(3, -1).numpy()
            valid_pred = pred_image.view(3, -1).numpy()

        print("==" * 10 + f"{_id}, {image_name}, {scene_id}" + "==" * 10)
        print("gt: ", valid_gt.shape, gt_image_path)
        print("pred: ", valid_pred.shape, pred_image_path)
        if mask is not None and not torch.all(mask):
            print("mask: ", mask.shape, mask_path)
        else:
            print("mask: not available")

        # Initialize color_corrector
        color_corrector = COLOR_CORRECTOR(
            method=cc_configs["method"],
            option=cc_configs["option"],
            batch_size=cc_configs["batch_size"],
            sample_size=cc_configs["sample_size"],
            mode=cc_configs["mode"])

        # Prepare for color_correction
        (train_Xs, train_Xt, eval_Xs, eval_Xt, eval_masks,
         eval_path2Is, eval_path2It, _) = color_corrector.preparation(
            path2trainIs=[pred_image_path],
            path2trainIt=[gt_image_path],
            path2trainmasks=[mask_path],
            path2evalIs=[pred_image_path],
            path2evalIt=[gt_image_path],
            path2evalmasks=[mask_path],
            sample_every=ALL_PIXELS // cc_configs["sample_size"],
            paired=True,
            offset=0,
            mode=cc_configs["mode"],
            train_batch_size=cc_configs["batch_size"],
            eval_batch_size=cc_configs["batch_size"],
            downsample=-1
        )

        # Color correction operator estimation
        color_corrector(Xs=train_Xs, Xt=train_Xt)

        # Apply color_correction
        transformed_Xs = color_corrector.transform_and_result(
            eval_Xs=eval_Xs,
            eval_Xt=eval_Xt,
            eval_masks=eval_masks,
            eval_batch_size=cc_configs["batch_size"]
        )

        transform_x = transformed_Xs[0]
        # post process
        eval_source_image_path = eval_path2Is[0]
        path_dict["pred"].append(str(eval_source_image_path))
        eval_target_image_path = eval_path2It[0]
        path_dict["gt"].append(str(eval_target_image_path))

        print("eval source (test DSLR): ", eval_source_image_path)
        print("eval target (test DSLR): ", eval_target_image_path)

        im1 = Image.open(eval_source_image_path).convert("RGB")

        im2 = Image.open(eval_target_image_path).convert("RGB")

        im3 = Image.fromarray(transform_x).convert("RGB")

        npim1 = np.asarray(im1).astype(np.uint8)
        npim2 = np.asarray(im2).astype(np.uint8)
        npim3 = np.asarray(im3).astype(np.uint8)
        print("transformed x: ", npim3.shape)

        if npim1.shape != npim2.shape:
            im1 = Image.fromarray(np.array(Image.fromarray(npim1).resize((npim2.shape[1], npim2.shape[0]))).astype(np.uint8)).convert("RGB")

        # save dir of color corrected img
        iphone_fname = Path(eval_source_image_path).stem
        nn_dslr_fname = Path(eval_target_image_path).stem
        assert iphone_fname == nn_dslr_fname

        print(collection_dir / Path(eval_source_image_path).name)
        cc_source_path = collection_dir / Path(eval_source_image_path).name
        path_dict["cc_pred"].append(str(cc_source_path))

        print("path_dict[pred]: ", path_dict["pred"])
        print("path_dict[cc_pred]: ", path_dict["cc_pred"])

        # save the color_corrected_pred -> pred_after_cc_dir
        im3.save(os.path.join(pred_after_cc_dir, Path(eval_source_image_path).name))

        # save the pair (before/after) -> collection_dir
        save_im_path = os.path.join(collection_dir, iphone_fname + ".png")
        concat_im = get_concat_h(im1=im1, im2=im3)
        concat_im.save(save_im_path)

    return path_dict


def color_correction_all(data_root, pred_dir, scene_list, upload_path, verbose=True):
    """
    Apply color correction to all scenes in the scene list.

    Args:
        data_root: Root directory containing scene data
        pred_dir: Directory containing predicted images
        scene_list: List of scene IDs to process
        upload_path: Path to upload directory
        verbose: Whether to print progress information

    Returns:
        dict: Dictionary mapping scene IDs to path dictionaries
    """
    path_dicts = {str(scene_id): None for scene_id in scene_list}
    for i, scene_id in enumerate(scene_list):
        if verbose:
            print(f"({i+1} / {len(scene_list)}) scene_id: {scene_id}")

        assert (
            Path(pred_dir) / scene_id
        ).exists(), f"Prediction dir of scene {scene_id} does not exist"
        num_images_pred = len(os.listdir(Path(pred_dir) / scene_id))
        assert num_images_pred > 0, f"Prediction dir of scene {scene_id} is empty"

        scene = ScannetppScene_Release(scene_id, data_root=data_root)

        # Get the list of test image names: "e.g., DSC09999.JPG"
        # but we need to refer the specific folder which contains DSLR after undistortion by iPhone intrinsic
        image_list = get_test_images(scene.dslr_nerfstudio_transform_path)  # read original transforms.json

        mask_dir = None
        mask_dir = scene.dslr_resized_mask_dir  # change to the DSLR_undistorted_iphone mask dir

        path_dict = color_correction(
            pred_dir=Path(pred_dir) / scene_id,
            gt_dir=scene.dslr_resized_dir,  # change to the DSLR_undistorted_iphone dir
            image_list=image_list,
            upload_path=upload_path,
            mask_dir=mask_dir,
            scene_id=scene_id,
            verbose=verbose,
        )

        path_dicts[scene_id] = path_dict

    return path_dicts


if __name__ == "__main__":
    data_root = ""
    pred_dir = ""
    scene_list = []
    upload_path = ""
