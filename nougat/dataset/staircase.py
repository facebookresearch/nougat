"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from collections import deque
import operator
import itertools
from typing import Optional, List, Tuple
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="All-NaN slice encountered")


def stair_func(x: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.heaviside(x[:, None] - np.floor(thresholds)[None, :], 0).sum(1)


def compute_gini(labels: np.ndarray) -> float:
    N = len(labels)
    if N == 0:
        return 0
    G = N - np.square(np.bincount(labels)).sum() / N
    return G


def compute_binary_gini(labels: np.ndarray) -> float:
    N = len(labels)
    if N == 0:
        return 0
    G = N - labels.sum() ** 2 / N
    return G


def gini_impurity(
    thresholds: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    classes: Optional[List[int]] = None,
    reduction: Optional[str] = "sum",
    padded: bool = True,
) -> float:
    """
    Calculate the Gini impurity of a dataset split on a set of thresholds.

    Args:
        thresholds (np.ndarray): The thresholds to split the data on.
        data (np.ndarray): The data to split.
        labels (np.ndarray): The labels for the data.
        classes (Optional[List[int]]): The classes to consider. If None, all classes are used.
        reduction (Optional[str]): The reduction to apply to the impurity. One of "none", "sum", or "mean".
        padded (bool): Whether to pad the thresholds with `[-0.5, data.max() + 0.5]`.

    Returns:
        float: The Gini impurity.
    """
    G = []
    if not padded:
        thresholds = np.insert(
            thresholds, [0, len(thresholds)], [-0.5, data.max() + 0.5]
        )
    if classes is None:
        classes = np.arange(len(thresholds) - 1)
    else:
        classes = np.asarray(classes)
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    masks = np.logical_and(
        data > thresholds[classes, None],
        data <= thresholds[classes + 1, None],
    )
    for i, c in enumerate(classes):
        G.append(compute_binary_gini(np.where(labels[masks[i]] == c, 1, 0)))

    if reduction is None or reduction == "none":
        return G
    elif reduction == "sum":
        return sum(G)
    elif reduction == "mean":
        return sum(G) / len(G)
    else:
        raise NotImplementedError


def step_impurity(
    thresholds,
    data: np.ndarray,
    labels: np.ndarray,
    classes: Optional[List[int]] = None,
) -> float:
    """
    Calculate the step-wise Gini impurity of a dataset split on a set of thresholds.

    Args:
        thresholds (np.ndarray): The thresholds to split the data on.
        data (np.ndarray): The data to split.
        labels (np.ndarray): The labels for the data.
        classes (Optional[List[int]]): The classes to consider. If None, all classes are used.

    Returns:
        float: The step-wise Gini impurity.
    """
    G = gini_impurity(thresholds, data, labels, reduction=None, classes=classes)
    out = []
    for i in range(len(G) - 1):
        out.append(G[i] + G[i + 1])
    return out


class PaddedArray:
    """
    A wrapper class for an array that allows for relative indexing.

    Args:
        array (np.ndarray): The array to wrap.
        range (Optional[Tuple[int, int]]): The range of the array to expose. Defaults to (1, -1).
    """

    def __init__(
        self, array: np.ndarray, range: Optional[Tuple[int, int]] = (1, -1)
    ) -> None:
        self.array = array
        mi, ma = range
        assert ma <= 0, "relative assignment only"
        self.range = mi, ma

    def __len__(self):
        return len(self.array) + self.range[1] - self.range[0]

    def _process_index(self, index):
        if isinstance(index, slice):
            index = slice(
                (index.start or 0) + self.range[0],
                self.range[0] + (len(self) if index.stop is None else index.stop),
                index.step,
            )
            if index.stop > len(self.array):
                raise IndexError
        else:
            index = index + self.range[0]
            if index > len(self):
                raise IndexError
        return index

    def __getitem__(self, index):
        index = self._process_index(index)
        return self.array[index]

    def __setitem__(self, index, value):
        self.array[self._process_index(index)] = value

    def copy(self):
        return PaddedArray(self.array.copy(), self.range)

    def toarray(self):
        return self.array[self.range[0] : self.range[1]]


class Staircase:
    """
    A class for learning a staircase decision tree.

    Args:
        domain: The number of points in the domain.
        n_classes: The number of classes.
    """

    def __init__(self, domain: int, n_classes: int) -> None:
        self.domain = domain
        self.classes = n_classes
        assert domain > 0
        assert n_classes > 0
        self.thresholds = self._back_thres = self._forward_thres = np.linspace(
            domain / n_classes, domain, n_classes - 1, endpoint=False
        )
        self.uncertainty = np.zeros_like(self.thresholds)

    def statistic_fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Fit statistical thresholds for anomaly detection.

        This method fits statistical thresholds for anomaly detection based on input data and labels.

        Args:
            data (np.ndarray): The input data.
            labels (np.ndarray): The labels corresponding to the data.

        Note:
            This method modifies the internal state of the object to set statistical thresholds.
        """
        onehot = np.eye(self.classes)[labels.reshape(-1)]
        onehot.reshape(list(labels.shape) + [self.classes])
        k = onehot * data.T.repeat(self.classes, 1)
        k[k == 0] = np.nan
        med = np.nanmedian(k, 0)
        for i in range(len(med)):
            if med[i] != med[i]:
                med[i] = 0 if i == 0 else med[i - 1]
        mad = 5 * np.nan_to_num(
            np.nanmedian(np.absolute(k - np.nanmedian(k, 0)), 0),
            nan=self.domain / self.classes / 2,
        )
        arr = np.vstack(((med - mad)[:-1], (med + mad)[1:]))
        self._forward_thres[:] = arr.max(0)
        self._back_thres[:] = arr.min(0)

        self._stat_forward = self._forward_thres.copy()
        self._stat_back = self._back_thres.copy()

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        early_stop_after: int = 10,
        fixed: bool = True,
    ) -> None:
        """
        Fit statistical thresholds for anomaly detection.

        This method fits statistical thresholds for anomaly detection based on input data and labels.

        Args:
            data (np.ndarray): The input data.
            labels (np.ndarray): The labels corresponding to the data.
            early_stop_after (int, optional): The number of consecutive early stops to consider. Default is 10.
            fixed (bool, optional): Whether to use fixed thresholds. Default is True.

        Note:
            This method modifies the internal state of the object to set statistical thresholds.
        """
        assert data.ndim == 1
        assert labels.ndim <= 2
        if self.classes == 1:
            self.thresholds = np.array([0.5 + data.max()])
            self.uncertainty = np.zeros_like(self.thresholds)
        if data.ndim == 1:
            data = np.expand_dims(data, 0)
        thresholds = PaddedArray(
            np.insert(
                np.arange(self.domain - self.classes + 1, self.domain) - 1,
                [0, self.classes - 1],
                [-0.5, self.domain + 0.5],
            ).astype(int)
        )
        self._back_thres = thresholds.copy()
        self._forward_thres = thresholds.copy()
        self.statistic_fit(data, labels)
        last = -0.5
        for n in range(self.classes):
            G = np.inf
            Gis = deque([], early_stop_after)
            # forward pass
            if n < self.classes - 1:
                new_forward_n: float = self._forward_thres[n]
                for i in range(
                    max(0, self._back_thres[n - 1]) if n - 1 >= 0 else int(last),
                    min(self.domain, self._forward_thres[n + 1])
                    if n + 2 < self.classes
                    else self.domain - 1,
                ):
                    thresholds.array[n + 1] = i + 0.5
                    Gi = step_impurity(
                        thresholds.array, data, labels, classes=[n, n + 1]
                    )[0]
                    Gis.append(Gi)
                    if Gi <= G:
                        last = i + 0.5
                        new_forward_n = last
                        G = Gi
                    elif (
                        (not fixed or i - last > self.domain / self.classes)
                        and len(Gis) == early_stop_after
                        and all(
                            itertools.starmap(
                                operator.ge,
                                zip(Gis, itertools.islice(Gis, 1, early_stop_after)),
                            )
                        )
                    ):
                        break
                thresholds.array[n + 1] = new_forward_n
                self._forward_thres.array[n + 1] = new_forward_n
                self._back_thres.array[n + 1] = new_forward_n
            G = np.inf
        self._forward_thres = self._forward_thres.toarray().clip(
            min=0, max=self.domain - 1
        )
        self._back_thres = self._back_thres.toarray().clip(min=0, max=self.domain - 1)
        self.thresholds = (self._forward_thres + self._back_thres) / 2
        self.uncertainty = np.abs(self._forward_thres - self._back_thres) / 2

    @property
    def score(self):
        try:
            return gini_impurity(self.thresholds, self._data, self._labels) / len(
                self._data
            )
        except AttributeError:
            return np.inf

    def predict(self, x: np.ndarray) -> np.ndarray:
        return stair_func(x, self.get_boundaries())

    def __call__(self, *args):
        return self.predict(*args)

    def get_boundaries(self) -> np.ndarray:
        return self.thresholds.astype(int).clip(min=0, max=self.domain - 1) + 0.5
