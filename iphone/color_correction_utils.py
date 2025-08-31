from typing import List, Optional
import numpy as np
import ot


class ColorCorrector:
    def __init__(self, method: str):
        self.method = method
        if method == "EmdTransport":
            self.system = ot.da.EMDTransport()
        elif method == "SinkhornTransport":
            self.system = ot.da.SinkhornTransport(reg_e=1e-1)
        elif method == "MappingTransportLinear":
            self.system = ot.da.MappingTransport(
                mu=1e0,
                eta=1e-8,
                bias=True,
                max_iter=10,
                verbose=True,
            )
        elif method == "MappingTransportGaussian":
            self.system = ot.da.MappingTransport(
                mu=1e0,
                eta=1e-2,
                sigma=1,
                bias=False,
                max_iter=5,
                verbose=True,
            )
        elif method == "LinearGWTransport":
            self.system = ot.da.LinearGWTransport()
        elif method == "LinearTransport":
            self.system = ot.da.LinearTransport()
        else:
            raise NotImplementedError(f"Option {method} not implemented.")

    def fit(
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
                sample_indices = np.arange(0, image_target.shape[0])
                # Random shuffle and select
                rng = np.random.default_rng(seed=seed)
                indices1 = rng.shuffle(sample_indices)[:self.sample_size]
                indices2 = rng.shuffle(sample_indices)[:self.sample_size]

                image_source = image_source[indices1]
                image_target = image_target[indices2]

            Xs.append(image_source)
            Xt.append(image_target)

        Xs = np.concatenate(Xs, axis=0)
        Xt = np.concatenate(Xt, axis=0)
        self.system.fit(Xs=Xs, Xt=Xt)

    def transform(
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
            Transformed image_source after color correction: [H, W, 3]
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
