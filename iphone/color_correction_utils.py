from typing import List, Optional, Union
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
import ot

from common.scene_release import ScannetppScene_Release
from eval.nvs import get_test_images


COLOR_MAGENTA = [1.0, 0.5, 1.0]     # The color that are masked
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


class ColorCorrector:
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

    def fit(self, Xs, Xt):
        """
        Fit the color correction system.

        Args:
            Xs: Source images array with shape [:, 3]
            Xt: Target images array with shape [:, 3]
        """
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
        else:
            raise NotImplementedError(f"Not implemented method: {self.method}")
        # elif self.method == "linear":
        #     assert self.option is not None
        #     if self.option == "SVD":
        #         self.system = Procrustes()
        #         self.system.fit(Xs=Xs, Xt=Xt)
        # elif self.method == "histogram":
        #     assert self.option == "scikit"
        #     self.system = Histogram()

    def prepare_and_fit(
        self,
        train_images_source: List[np.ndarray],
        train_images_target: List[np.ndarray],
        train_image_masks: List[Optional[np.ndarray]],
        paired: bool = True,
        sample_every: int = 4,
        seed: int = 42,
    ):
        """
        Prepare training data and fit the color correction system.

        Args:
            train_images_source: List of source images (iPhone)
            train_images_target: List of target images (DSLR)
            train_image_masks: List of masks for the images
        """
        Xs = []
        Xt = []
        assert len(train_images_source) == len(train_images_target) == len(train_image_masks), "Training data lists must have the same length."
        num_images = len(train_images_source)
        for image_idx in range(num_images):
            image_source = train_images_source[image_idx]
            image_target = train_images_target[image_idx]
            image_mask = train_image_masks[image_idx]

            assert image_source.shape == image_target.shape, f"Image shapes do not match: {image_source.shape}, {image_target.shape}"
            assert image_source.shape[2] == 3, f"Image shape is not correct: {image_source.shape}. Should be (H, W, 3)."
            assert len(image_source.shape) == 3, f"Image shape is not correct: {image_source.shape}. Should be (H, W, 3)."

            image_source = image_source.reshape(-1, 3)
            image_target = image_target.reshape(-1, 3)

            if image_mask is not None:
                image_mask = image_mask.reshape(-1).astype(bool)
                image_source = image_source[image_mask]
                image_target = image_target[image_mask]

            if paired:
                # Uniformly sampled pixels on both source and target images
                sample_indices = np.arange(0, image_target.shape[0], sample_every)
                image_source = image_source[sample_indices]
                image_target = image_target[sample_indices]
            else:
                # Sample random pixels on both source and target images
                rng = np.random.default_rng(seed=seed)
                # Random shuffle and select
                sample_indices = np.arange(0, image_target.shape[0])
                indices1 = rng.shuffle(sample_indices)[:self.sample_size]
                indices2 = rng.shuffle(sample_indices)[:self.sample_size]

                image_source = image_source[indices1]
                image_target = image_target[indices2]
                # idx1 = rng.integers(0, image_source.shape[0], size=self.sample_size)
                # idx2 = rng.integers(0, image_target.shape[0], size=self.sample_size)

            Xs.append(image_source)
            Xt.append(image_target)

        Xs = np.concatenate(Xs, axis=0)
        Xt = np.concatenate(Xt, axis=0)
        self.fit(Xs=Xs, Xt=Xt)

    def transform_and_result(
        self,
        image_source: np.ndarray,
        image_target: np.ndarray,
        image_mask: Optional[np.ndarray] = None,
    ):
        """
        Apply color correction transformation.
        Args:
            image_source, image_target: [H, W, 3]
            image_mask: [H, W]
        Returns:
        """
        assert image_source.shape == image_target.shape, f"Input shapes do not match: {image_source.shape}, {image_target.shape}"
        assert image_source.shape[2] == 3, f"Input shape is not correct: {image_source.shape}. . Should be (H, W, 3)."
        assert len(image_source.shape) == 3, f"Input shape is not correct: {image_source.shape}. Should be (H, W, 3)."
        assert self.system is not None, "Color correction system is not initialized. Please call fit() first."

        image_shape = image_source.shape
        image_source_cc = self.system.transform(Xs=image_source.reshape(-1, 3))
        image_source_cc = image_source_cc.reshape(image_shape)
        image_source_cc = np.clip(image_source_cc, 0, 1)
        return image_source_cc
        # l2_loss_before = np.linalg.norm(eval_Xs - eval_Xt, axis=1).mean()
        # l2_loss_after = np.linalg.norm(eval_Xs_transformed - eval_Xt, axis=1).mean()
        # out = self.mat2im(eval_Xs_transformed, self.image_shape)
        # out = self.minmax(out)
        # return out

    # def transform_and_result2(
    #     self,
    #     eval_Xs,
    #     eval_Xt,
    #     eval_masks,
    #     eval_batch_size=1,
    # ):
    #     """
    #     Apply color correction transformation.

    #     Args:
    #         eval_Xs: Evaluation source images
    #         eval_Xt: Evaluation target images
    #         eval_masks: Evaluation masks
    #         eval_batch_size: Batch size for evaluation

    #     Returns:
    #         List of transformed images
    #     """
    #     # iPhone (source to be color-corrected)
    #     Xs = np.array(eval_Xs).reshape(eval_batch_size, self.all_sample_size, 3)
    #     print(eval_Xs.shape, eval_Xt.shape, Xs.shape)
    #     # DSLR (source to be used for estimating cc-function) - not used in current implementation
    #     # Xt = np.array(eval_Xt).reshape(eval_batch_size, self.all_sample_size, 3)
    #     # iPhone, magenta == 1.0
    #     eval_masks = np.array(eval_masks)

    #     transformed_Xs = []
    #     for i in range(eval_batch_size):
    #         mask = eval_masks[i]  # already image shape

    #         # if self.method == "histogram":
    #         #     image = self.mat2im(Xs[i], self.image_shape)
    #         #     reference = self.mat2im(Xt[i], self.image_shape)
    #         #     out = match_histograms(image, reference, channel_axis=-1)
    #         #     print("out red: ", out[:, :, 0].min(), out[:, :, 0].max())
    #         #     print("out blue: ", out[:, :, 1].min(), out[:, :, 1].max())
    #         #     print("out green: ", out[:, :, 2].min(), out[:, :, 2].max())
    #         #     _out = self.minmax(out)
    #         # else:
    #         X1 = Xs[i]  # [H*W, 3]
    #         print("origin", np.linalg.norm(eval_Xs - X1.reshape(-1, 3), axis=1).mean())
    #         print("before", np.linalg.norm(eval_Xt - eval_Xs, axis=1).mean())
    #         # print(np.sum(np.power(X1 - eval_Xs[i], 2)))
    #         # print(np.sum(np.power(X1 - eval_Xt[i], 2)))
    #         # print(X1.shape, X1.min(), X1.max())
    #         X1 = self.system.transform(Xs=X1)
    #         print("after", np.linalg.norm(eval_Xt - X1.reshape(-1, 3), axis=1).mean())
    #         breakpoint()
    #         # print(np.sum(np.power(X1 - eval_Xt[i], 2)))
    #         # print(X1.shape, X1.min(), X1.max())
    #         out = self.minmax(
    #             self.mat2im(
    #                 X1,
    #                 self.image_shape,
    #             )
    #         )
    #         print("out: ", out.shape)
    #         out[mask] = COLOR_MAGENTA
    #         transformed_Xs.append((out * 255).astype(np.uint8))
    #     return transformed_Xs

    def preparation(
        self,
        train_images_path_list: List[Union[str, Path]] = [],
        train_gt_path_list: List[Union[str, Path]] = [],
        train_masks_path_list: List[Union[str, Path]] = [],
        eval_images_path_list: List[Union[str, Path]] = [],
        eval_gt_path_list: List[Union[str, Path]] = [],
        eval_masks_path_list: List[Union[str, Path]] = [],
        sample_every: int = 4,
        paired: bool = True,
        offset: int = 0,
        mode: str = "",
        train_batch_size: int = 10,
        eval_batch_size: int = 10,
        downsample: int = -1,
        seed: int = 42,
    ):
        assert offset == 0
        assert train_batch_size == 1
        assert eval_batch_size == 1
        """
        Prepare training and evaluation data for color correction.

        Args:
            train_images_path_list: List of training iPhone image paths
            train_gt_path_list: List of training DSLR image paths
            train_masks_path_list: List of training mask paths
            eval_images_path_list: List of evaluation iPhone image paths
            eval_gt_path_list: List of evaluation DSLR image paths
            eval_masks_path_list: List of evaluation mask paths
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
        # if train_images_path_list is None:
        #     train_images_path_list = []
        # if train_gt_path_list is None:
        #     train_gt_path_list = []
        # if train_masks_path_list is None:
        #     train_masks_path_list = []
        # if eval_images_path_list is None:
        #     eval_images_path_list = []
        # if eval_gt_path_list is None:
        #     eval_gt_path_list = []
        # if eval_masks_path_list is None:
        #     eval_masks_path_list = []

        Xs = None
        Xt = None
        # Xs = []
        # Xt = []

        print("train_images_path_list: ", len(train_images_path_list))
        print("train_gt_path_list: ", len(train_gt_path_list))
        print("sample_size: ", self.sample_size, ALL_PIXELS // sample_every)
        all_samples = len(train_images_path_list)
        print("all samples: ", all_samples)

        # train_pairs_save_dir = os.path.join(os.path.dirname(train_images_path_list[0]), "train_pairs")
        # if not os.path.exists(train_pairs_save_dir):
        #     os.makedirs(train_pairs_save_dir)
        for image_idx in range(offset, offset + train_batch_size):
            """
            Process training data:
            - Xs: train source images *applied mask (#sample_size*train_batch_size, 3)
            - Xt: train target images *applied mask (#sample_size*train_batch_size, 3)
            """
            print("---train---")
            path2is = train_images_path_list[image_idx]  # iPhone
            # TODO: ??????
            # if str(path2is)[-10] == "_":
            #     path2is = Path(str(path2is)[:-9] + "0" + str(path2is)[-9:])

            path2it = train_gt_path_list[image_idx]  # DSLR
            path2mask = train_masks_path_list[image_idx]  # common mask
            has_diff_imsize = False

            # I1 = plt.imread(path2is).astype(np.float64) / 255  # iPhone
            # I2 = plt.imread(path2it).astype(np.float64) / 255  # DSLR
            I1 = Image.open(path2is).convert("RGB")  # iPhone
            I2 = Image.open(path2it).convert("RGB")  # DSLR
            # save_im_path = os.path.join(
            #     train_pairs_save_dir,
            #     f"{Path(path2is).stem}_{Path(path2it).stem}.png",
            # )

            # if I1.shape != I2.shape:
            #     has_diff_imsize = True
            #     I1 = np.array(
            #         Image.open(path2is).convert("RGB").resize((I2.size[1], I2.size[0]))
            #     ).astype(np.float64)
            #     I1 = I1 / 255
            #     print("I1 shape: ", I1.shape)
            #     print("I2 shape: ", I2.shape)
            if I1.size != I2.size:
                has_diff_imsize = True
                I1 = I1.resize(I2.size, Image.BICUBIC)
                print("I1 shape: ", I1.size)
                print("I2 shape: ", I2.size)

            I1 = np.array(I1).astype(np.float64) / 255
            I2 = np.array(I2).astype(np.float64) / 255
            w = I2.shape[1]
            h = I2.shape[0]
            print("I1 shape: ", I1.shape)
            print("I2 shape: ", I2.shape)
            # if downsample > 0:
            #     if not has_diff_imsize:
            #         I1 = Image.open(path2is)

            #     I1 = np.array(I1.resize((w // downsample, h // downsample))).astype(np.float64)
            #     I1 = I1 / 255
            #     downsampled_w = w // downsample
            #     downsampled_h = h // downsample
            #     I2 = Image.open(path2it)
            #     I2 = np.array(I2.resize((w // downsample, h // downsample))).astype(np.float64)
            #     I2 = I2 / 255

            #     print("I1 downsampled: ", I1.shape)
            #     print("I2 downsampled: ", I2.shape)

            # concat_train_pair = get_concat_h(
            #     im1=Image.fromarray(I1.astype(np.uint8)).convert("RGB"),
            #     im2=Image.fromarray(I2.astype(np.uint8)).convert("RGB"),
            #     im3=None
            # )
            # concat_train_pair.save(save_im_path)

            _Xs = self.im2mat(I1)
            _Xt = self.im2mat(I2)

            if os.path.exists(path2mask):
                # mask = plt.imread(path2mask).astype(bool)  # [H, W, 1]
                mask = np.array(Image.open(path2mask).convert("L")).astype(bool)    # (H, W)
                print(mask.shape)
                # if downsample > 0:
                #     mask = np.array(Image.open(path2mask).resize(
                #         (downsampled_w, downsampled_h))).astype(bool)
                # print("downsampled mask: ", mask.shape)
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

                # if _Xs.shape[0] <= _Xt.shape[0]:
                #     idx = np.arange(0, _Xs.shape[0], sample_every)
                # else:
                sample_indices = np.arange(0, _Xt.shape[0], sample_every)

                if image_idx == 0 or mode == "server":
                    Xs = np.array(_Xs[sample_indices, :]).astype(np.float64).reshape(-1, 3)
                    Xt = np.array(_Xt[sample_indices, :]).astype(np.float64).reshape(-1, 3)
                else:

                    Xs = np.concatenate((Xs, np.array(_Xs[sample_indices, :]).astype(np.float64).reshape(-1, 3)), axis=0)
                    Xt = np.concatenate((Xt, np.array(_Xt[sample_indices, :]).astype(np.float64).reshape(-1, 3)), axis=0)
                print("actual sampled pixels: ", len(sample_indices))
            else:
                print("unpaired random pixel selections")

                # Note: rng is not defined in this scope - this would need to be fixed
                # idx1 = rng.randint(_Xs.shape[0], size=(self.sample_size,))
                # idx2 = rng.randint(_Xt.shape[0], size=(self.sample_size,))
                rng = np.random.default_rng(seed=seed)
                idx1 = rng.integers(0, _Xs.shape[0], size=self.sample_size)
                idx2 = rng.integers(0, _Xt.shape[0], size=self.sample_size)

                if image_idx == 0 or mode == "server":
                    Xs = np.array(_Xs[idx1, :]).astype(np.float64).reshape(-1, 3)
                    Xt = np.array(_Xt[idx2, :]).astype(np.float64).reshape(-1, 3)
                else:
                    Xs = np.concatenate((Xs, np.array(_Xs[idx1, :]).astype(np.float64).reshape(-1, 3)), axis=0)
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

        for image_idx in range(offset, offset + eval_batch_size):  # eval_batch_size
            """
            Process evaluation data:
            - eval_Xs: eval source images (#sample_size*eval_batch_size, 3)
            - eval_Xt: train target images (#sample_size*eval_batch_size, 3)
            - eval_masks: inverted eval masks (H, W), magenta=True, else False
            """
            print("---eval---")
            print("index: ", image_idx)

            path2is = eval_images_path_list[image_idx]  # source
            path2it = eval_gt_path_list[image_idx]  # target
            path2mask = eval_masks_path_list[image_idx]

            # I1 = plt.imread(path2is).astype(np.float64) / 255  # iPhone
            # I2 = plt.imread(path2it).astype(np.float64) / 255  # DSLR

            # if I1.shape != I2.shape:
            #     print("I1 shape: ", I1.shape)
            #     print("I2 shape: ", I2.shape)
            #     I1 = np.array(Image.open(path2is).resize((I2.shape[1], I2.shape[0]))).astype(np.uint8)
            #     print("I1 shape: ", I1.shape)
            #     print("I2 shape: ", I2.shape)

            I1 = Image.open(path2is).convert("RGB")  # iPhone
            I2 = Image.open(path2it).convert("RGB")  # DSLR
            if I1.size != I2.size:
                I1 = I1.resize(I2.size, Image.BICUBIC)
                print("I1 shape: ", I1.size)
                print("I2 shape: ", I2.size)
            I1 = np.array(I1).astype(np.float64) / 255
            I2 = np.array(I2).astype(np.float64) / 255

            # if self.image_shape is None:
            #     self.image_shape = I1.shape

            _Xs = self.im2mat(I1)
            _Xt = self.im2mat(I2)

            if self.all_sample_size is None:
                self.all_sample_size = _Xs.shape[0]
                print("all sample size: ", self.all_sample_size)

            if os.path.exists(path2mask):
                # mask = plt.imread(path2mask).astype(bool)  # [H, W, 1]
                mask = np.array(Image.open(path2mask).convert("L")).astype(bool)    # (H, W)
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

        return (
            Xs,
            Xt,
            eval_Xs,
            eval_Xt,
            eval_masks,
            eval_path2Is,
            eval_path2It,
            # train_pairs_save_dir,
            None,
        )  # [self.batch_size, -1, 3]

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
