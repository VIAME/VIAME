#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Descriptor indexing for VIAME video search (viame.core.index_descriptors).

Implements ITQ (Iterative Quantization) locality-sensitive hashing with only
numpy, the descriptor sources it can be trained from (CSV, KWIVER descriptor
CSV, PostgreSQL), and the per-video file bundles that make up the
file-backed search index. The command line lives in the ``viame index``
applet (tools/index.py); this module is import-only.

The ITQ algorithm:
1. Optionally normalizes input descriptors
2. Centers the data by subtracting the mean
3. Projects data to bit_length dimensions using PCA
4. Iteratively refines a rotation matrix to minimize quantization error
5. Produces binary hash codes for efficient similarity search

References:
    Gong, Y., & Lazebnik, S. (2011). Iterative quantization: A procrustean approach
    to learning binary codes. In CVPR.
    http://www.cs.unc.edu/~lazebnik/publications/cvpr11_small_code.pdf
"""

import json
import os
import time
import pickle
import sys
from collections import defaultdict

import numpy as np


class ITQModel:
    """
    Iterative Quantization (ITQ) model for locality-sensitive hashing.

    This class implements the ITQ algorithm for learning binary hash codes
    from high-dimensional descriptor vectors.
    """

    # PCA method options
    PCA_COV_EIG = 'cov_eig'      # Covariance + eigendecomposition (default)
    PCA_DIRECT_SVD = 'direct_svd'  # Direct SVD on centered data (more stable)

    # Rotation initialization options
    INIT_SVD = 'svd'    # SVD orthogonalization (default)
    INIT_QR = 'qr'      # QR decomposition

    def __init__(self, bit_length=256, itq_iterations=100, random_seed=0,
                 normalize=None, pca_method='cov_eig', init_method='svd'):
        """
        Initialize ITQ model parameters.

        Default values are for the standard ITQ configuration.

        Args:
            bit_length: Number of bits in the hash code (default: 256)
            itq_iterations: Number of ITQ refinement iterations (default: 100)
            random_seed: Random seed for reproducibility (default: 0)
            normalize: Normalization order for input vectors (default: None = no normalization).
                       Can be any valid numpy.linalg.norm 'ord' parameter (e.g., 2 for L2 norm).
            pca_method: PCA computation method (default: 'cov_eig')
                       - 'cov_eig': Covariance matrix + eigendecomposition (default)
                       - 'direct_svd': Direct SVD on centered data (more numerically stable)
            init_method: Rotation matrix initialization method (default: 'svd')
                        - 'svd': SVD orthogonalization (default)
                        - 'qr': QR decomposition
        """
        self.bit_length = bit_length
        self.itq_iterations = itq_iterations
        self.random_seed = random_seed
        self.normalize = normalize
        self.pca_method = pca_method
        self.init_method = init_method

        # Model parameters (learned during training)
        self.mean_vec = None
        self.rotation = None

        # Validate normalization parameter
        if normalize is not None:
            self._norm_vector(np.random.rand(8))

    def _norm_vector(self, v):
        """
        Normalize vector(s) using configured normalization order.

        Args:
            v: Input vector or matrix (if matrix, normalizes along last axis)

        Returns:
            Normalized vector/matrix, or original if normalize is None
        """
        if self.normalize is not None:
            n = np.linalg.norm(v, self.normalize, v.ndim - 1, keepdims=True)
            # Replace 0's with 1's to prevent division by zero
            n[n == 0.] = 1.
            return v / n
        return v

    def _find_itq_rotation(self, v, n_iter, verbose=False, report_interval=1.0):
        """
        Find optimal rotation matrix using ITQ algorithm.

        Args:
            v: PCA-projected data, shape (n_samples, bit_length)
            n_iter: Number of ITQ iterations
            verbose: Print progress messages
            report_interval: Seconds between progress reports (default: 1.0)

        Returns:
            Tuple of (binary_codes, rotation_matrix)
        """
        import time

        bit = v.shape[1]

        # Initialize with orthogonal random rotation using SVD (default)
        # or QR decomposition (alternative)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        r = np.random.randn(bit, bit)

        if self.init_method == self.INIT_SVD:
            # SVD orthogonalization
            u11, s2, v2 = np.linalg.svd(r)
            r = u11[:, :bit]
        else:  # INIT_QR
            # Alternative: QR decomposition
            r, _ = np.linalg.qr(r)

        # ITQ iterations to find optimal rotation
        if verbose:
            print(f"    Running {n_iter} ITQ iterations...")

        last_report_time = time.time()

        for i in range(n_iter):
            # Rotate data
            z = np.dot(v, r)

            # Compute binary codes using sign function
            # Uses -1/+1 representation during iteration
            ux = np.ones(z.shape) * (-1)
            ux[z >= 0] = 1

            # Update rotation matrix using Orthogonal Procrustes
            c = np.dot(ux.transpose(), v)
            ub, sigma, ua = np.linalg.svd(c)
            r = np.dot(ua, ub.transpose())

            # Progress reporting
            current_time = time.time()
            if verbose and (current_time - last_report_time >= report_interval):
                z_new = np.dot(v, r)
                quantization_error = np.mean((ux - z_new) ** 2)
                print(f"      Iteration {i + 1}/{n_iter}, "
                      f"quantization error: {quantization_error:.6f}")
                last_report_time = current_time

        # Compute final binary codes with the final rotation matrix
        # This ensures b and r are synchronized
        z = np.dot(v, r)
        b = np.zeros(z.shape, dtype=np.bool_)
        b[z >= 0] = True

        return b, r

    def fit(self, descriptors, verbose=True, report_interval=1.0):
        """
        Train the ITQ model on a set of descriptors.

        Args:
            descriptors: numpy array of shape (n_samples, n_features)
            verbose: Print progress messages (default: True)
            report_interval: Seconds between progress reports (default: 1.0)

        Returns:
            Binary hash codes for the training descriptors
        """
        n_samples, n_features = descriptors.shape

        if n_features < self.bit_length:
            raise ValueError(
                f"Input descriptors have fewer features ({n_features}) than "
                f"requested bit encoding ({self.bit_length}). Hash codes will be "
                "smaller than requested due to PCA decomposition result being "
                "bound by number of features."
            )

        if verbose:
            print(f"  Training ITQ model on {n_samples} descriptors "
                  f"({n_features} dimensions)")

        # Step 1: Normalize descriptors if configured
        if verbose and self.normalize is not None:
            print(f"    Normalizing descriptors (ord={self.normalize})...")
        x = self._norm_vector(descriptors.astype(np.float64))

        # Step 2: Center data
        if verbose:
            print("    Computing mean vector and centering data...")
        self.mean_vec = np.mean(x, axis=0)
        x = x - self.mean_vec

        # Step 3: PCA transformation
        if verbose:
            print(f"    Computing PCA transformation (method: {self.pca_method})...")

        if self.pca_method == self.PCA_COV_EIG:
            # Covariance matrix + eigendecomposition
            if verbose:
                print("      Computing covariance matrix...")
            # numpy.cov expects features as rows, observations as columns
            c = np.cov(x.transpose())

            if verbose:
                print("      Computing eigendecomposition...")
            # Get eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(c)

            # Sort by descending eigenvalue magnitude
            if verbose:
                print("      Ordering eigenvectors by descending eigenvalue...")
            sorted_indices = np.argsort(eigenvalues)[::-1]

            # Keep top bit_length eigenvectors (as columns)
            pc_top = eigenvectors[:, sorted_indices[:self.bit_length]]

            # Handle complex eigenvalues (take real part)
            if np.iscomplexobj(pc_top):
                pc_top = pc_top.real

        else:  # PCA_DIRECT_SVD
            # Alternative: Direct SVD on centered data (more numerically stable)
            if verbose:
                print("      Computing SVD...")
            U, S, Vt = np.linalg.svd(x, full_matrices=False)
            # PCA components are rows of Vt, take top bit_length
            pc_top = Vt[:self.bit_length, :].T

        # Project data onto principal components
        if verbose:
            print("      Projecting data to reduced dimensions...")
        v = np.dot(x, pc_top)

        # Step 4: ITQ to find optimal rotation
        if verbose:
            print("    Performing ITQ to find optimal rotation...")
        binary_codes, itq_rotation = self._find_itq_rotation(
            v, self.itq_iterations, verbose=verbose, report_interval=report_interval
        )

        # Combine PCA projection with ITQ rotation
        # This allows single-step projection during hash computation
        self.rotation = np.dot(pc_top, itq_rotation)

        if verbose:
            print("    ITQ training complete")

        return binary_codes

    def get_hash(self, descriptor):
        """
        Compute hash code for a single descriptor.

        Args:
            descriptor: 1D numpy array of shape (n_features,)

        Returns:
            Binary hash code as numpy boolean array
        """
        z = np.dot(self._norm_vector(descriptor) - self.mean_vec, self.rotation)
        b = np.zeros(z.shape, dtype=bool)
        b[z >= 0] = True
        return b

    def get_hash_bytes(self, descriptor):
        """
        Compute hash code for a single descriptor, returned as bytes.

        Args:
            descriptor: 1D numpy array of shape (n_features,)

        Returns:
            Binary hash code as bytes
        """
        return self._pack_bits(self.get_hash(descriptor))

    def compute_hashes(self, descriptors):
        """
        Compute hash codes for multiple descriptors.

        Args:
            descriptors: numpy array of shape (n_samples, n_features)

        Returns:
            List of binary hash codes as bytes
        """
        # Normalize, center, and project all at once
        normalized = self._norm_vector(descriptors)
        centered = normalized - self.mean_vec
        projected = np.dot(centered, self.rotation)

        # Quantize to binary
        binary = (projected >= 0)

        # Pack each row into bytes
        return [self._pack_bits(row) for row in binary]

    def compute_hashes_bool(self, descriptors):
        """
        Compute hash codes for multiple descriptors as boolean arrays.

        Args:
            descriptors: numpy array of shape (n_samples, n_features)

        Returns:
            numpy boolean array of shape (n_samples, bit_length)
        """
        normalized = self._norm_vector(descriptors)
        centered = normalized - self.mean_vec
        projected = np.dot(centered, self.rotation)
        return projected >= 0

    def _pack_bits(self, bits):
        """
        Pack binary array into bytes.

        Args:
            bits: 1D numpy array of booleans or 0s and 1s

        Returns:
            bytes object
        """
        bits_uint8 = np.asarray(bits, dtype=np.uint8)
        # Pad to multiple of 8
        n_bits = len(bits_uint8)
        n_bytes = (n_bits + 7) // 8
        padded = np.zeros(n_bytes * 8, dtype=np.uint8)
        padded[:n_bits] = bits_uint8

        # Pack into bytes
        byte_array = np.packbits(padded)
        return bytes(byte_array)

    def save(self, output_dir, prefix="itq.model"):
        """
        Save model parameters to numpy files.

        Args:
            output_dir: Directory to save model files
            prefix: Filename prefix (default: "itq.model")

        Returns:
            Tuple of (mean_vec_path, rotation_path)
        """
        os.makedirs(output_dir, exist_ok=True)

        # Use standard naming convention
        r_str = self.random_seed if self.random_seed is not None else 0
        suffix = f"b{self.bit_length}_i{self.itq_iterations}_r{r_str}"

        mean_path = os.path.join(output_dir, f"{prefix}.{suffix}.mean_vec.npy")
        rotation_path = os.path.join(output_dir, f"{prefix}.{suffix}.rotation.npy")

        np.save(mean_path, self.mean_vec)
        np.save(rotation_path, self.rotation)

        return mean_path, rotation_path

    def load(self, output_dir, prefix="itq.model"):
        """
        Load model parameters from numpy files.

        Args:
            output_dir: Directory containing model files
            prefix: Filename prefix (default: "itq.model")

        Returns:
            self
        """
        r_str = self.random_seed if self.random_seed is not None else 0
        suffix = f"b{self.bit_length}_i{self.itq_iterations}_r{r_str}"

        mean_path = os.path.join(output_dir, f"{prefix}.{suffix}.mean_vec.npy")
        rotation_path = os.path.join(output_dir, f"{prefix}.{suffix}.rotation.npy")

        self.mean_vec = np.load(mean_path)
        self.rotation = np.load(rotation_path)

        return self

    def has_model(self):
        """Check if model parameters are loaded."""
        return self.mean_vec is not None and self.rotation is not None


class DescriptorSource:
    """Base class for descriptor data sources."""

    def get_descriptors(self, max_count=None, uids=None, random_sample=False):
        """
        Retrieve descriptors.

        Args:
            max_count: Maximum number of descriptors to retrieve (None = all)
            uids: Specific UIDs to retrieve (None = all)
            random_sample: If True and max_count < total, randomly sample instead
                          of taking the first max_count (default: False)

        Returns:
            Tuple of (uids_list, descriptors_array)
        """
        raise NotImplementedError

    def get_all_uids(self):
        """Get all available UIDs."""
        raise NotImplementedError

    def __len__(self):
        """Return total number of descriptors available."""
        raise NotImplementedError


class CSVDescriptorSource(DescriptorSource):
    """Load descriptors from CSV file."""

    def __init__(self, file_path):
        """
        Initialize CSV descriptor source.

        Args:
            file_path: Path to CSV file (format: uid,val1,val2,...,valN)
        """
        self.file_path = file_path
        self._cache = None

    def _load_all(self):
        """Load and cache all descriptors from file."""
        if self._cache is not None:
            return self._cache

        uids = []
        descriptors = []

        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) < 2:
                    continue

                uid = parts[0]
                try:
                    values = [float(x) for x in parts[1:]]
                    if values:
                        uids.append(uid)
                        descriptors.append(values)
                except ValueError:
                    continue

        self._cache = (uids, np.array(descriptors) if descriptors else np.array([]))
        return self._cache

    def get_descriptors(self, max_count=None, uids=None, random_sample=False):
        all_uids, all_descs = self._load_all()

        if len(all_descs) == 0:
            return [], np.array([])

        if uids is not None:
            uid_set = set(uids)
            indices = [i for i, u in enumerate(all_uids) if u in uid_set]
            return [all_uids[i] for i in indices], all_descs[indices]

        if max_count is not None and max_count < len(all_uids):
            if random_sample:
                # Random subsampling
                indices = np.random.choice(len(all_uids), max_count, replace=False)
                return [all_uids[i] for i in indices], all_descs[indices]
            else:
                return all_uids[:max_count], all_descs[:max_count]

        return all_uids, all_descs

    def get_all_uids(self):
        all_uids, _ = self._load_all()
        return all_uids

    def __len__(self):
        all_uids, _ = self._load_all()
        return len(all_uids)


class KwiverCsvDescriptorSource(DescriptorSource):
    """Load descriptors from a KWIVER track-descriptor CSV.

    This is the file the index pipelines' ``write_track_descriptor`` (csv
    writer) produces, one descriptor per line::

        uid, type, n_track_refs, "id id ...", n_values, "v1 v2 ...", n_hist, "..."

    Lines whose value count is 0 carry no vector (they were stripped after
    the bundle was built) and are skipped.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self._cache = None

    @staticmethod
    def parse_line(line):
        """Return (uid, values) for one data line, or None."""
        parts = line.split(',')
        if len(parts) != 8:
            return None
        uid = parts[0].strip()
        try:
            count = int(parts[4].strip() or 0)
        except ValueError:
            return None
        if count <= 0:
            return uid, None
        try:
            values = [float(x) for x in parts[5].split()]
        except ValueError:
            return None
        if len(values) != count:
            return None
        return uid, values

    def _load_all(self):
        if self._cache is not None:
            return self._cache
        uids = []
        descriptors = []
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parsed = self.parse_line(line)
                if parsed is None or parsed[1] is None:
                    continue
                uids.append(parsed[0])
                descriptors.append(parsed[1])
        self._cache = (uids, np.array(descriptors, dtype=np.float32)
                       if descriptors else np.array([], dtype=np.float32))
        return self._cache

    def get_descriptors(self, max_count=None, uids=None, random_sample=False):
        all_uids, all_descs = self._load_all()
        if len(all_descs) == 0:
            return [], np.array([])
        if uids is not None:
            uid_set = set(uids)
            indices = [i for i, u in enumerate(all_uids) if u in uid_set]
            return [all_uids[i] for i in indices], all_descs[indices]
        if max_count is not None and max_count < len(all_uids):
            if random_sample:
                indices = np.random.choice(len(all_uids), max_count, replace=False)
                return [all_uids[i] for i in indices], all_descs[indices]
            return all_uids[:max_count], all_descs[:max_count]
        return all_uids, all_descs

    def get_all_uids(self):
        return self._load_all()[0]

    def __len__(self):
        return len(self._load_all()[0])


# ---------------------------------------------------------------------------
# File-backed index bundles
#
# One set of files per indexed video, all in the index ("database") folder
# and sharing the video's basename:
#
#   <name>.index             JSON manifest; its presence marks an indexed video
#   <name>_descriptors.csv   track descriptors (uids, track refs, history;
#                            raw vectors until the bundle is built)
#   <name>_tracks.csv        object tracks referenced by the descriptors
#   <name>_descriptors.npy   N x D float32 descriptor matrix
#   <name>_uids.txt          uid of each row of the matrix
#   <name>_hashes.npy        N x bits uint8 ITQ codes (0/1) of each row
#   ITQ/itq.model.*.npy      the one ITQ model shared by every bundle
# ---------------------------------------------------------------------------
BUNDLE_MANIFEST_VERSION = 1
BUNDLE_INDEX_POSTFIX = ".index"
BUNDLE_DESCRIPTOR_CSV_POSTFIX = "_descriptors.csv"
BUNDLE_DESCRIPTOR_NPY_POSTFIX = "_descriptors.npy"
BUNDLE_UIDS_POSTFIX = "_uids.txt"
BUNDLE_HASHES_POSTFIX = "_hashes.npy"


def _model_suffix(bit_length, itq_iterations, random_seed):
    return "b%d_i%d_r%d" % (bit_length, itq_iterations,
                            random_seed if random_seed is not None else 0)


def _model_hash(itq_dir, suffix):
    """Short digest of the model files, stored in manifests so bundles hashed
    with an older model are recognised and rehashed."""
    import hashlib
    digest = hashlib.sha1()
    for kind in ("mean_vec", "rotation"):
        path = os.path.join(itq_dir, "itq.model.%s.%s.npy" % (suffix, kind))
        with open(path, 'rb') as f:
            digest.update(f.read())
    return digest.hexdigest()[:12]


def _model_present(itq_dir, suffix):
    return all(os.path.exists(os.path.join(itq_dir, "itq.model.%s.%s.npy" % (suffix, kind)))
               for kind in ("mean_vec", "rotation"))


def detect_backend(database_dir):
    """How an index folder stores descriptors: "postgres" when it holds an
    embedded server (SQL/) and no file bundles, "files" otherwise (including
    a folder that does not exist yet)."""
    if not os.path.isdir(database_dir):
        return "files"
    has_sql = os.path.isdir(os.path.join(database_dir, "SQL"))
    return "postgres" if has_sql and not list_index_bundles(database_dir) else "files"


def list_index_bundles(database_dir):
    """Basenames of every video with a descriptor CSV or array in the folder."""
    names = set()
    for filename in os.listdir(database_dir):
        for postfix in (BUNDLE_DESCRIPTOR_CSV_POSTFIX, BUNDLE_DESCRIPTOR_NPY_POSTFIX):
            if filename.endswith(postfix):
                names.add(filename[:-len(postfix)])
    return sorted(names)


def read_bundle_manifest(database_dir, name):
    path = os.path.join(database_dir, name + BUNDLE_INDEX_POSTFIX)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            manifest = json.load(f)
        return manifest if isinstance(manifest, dict) else None
    except (OSError, ValueError):
        return None


def _write_atomic(path, writer):
    """Write through a temp file and rename, so a crash never leaves a
    truncated array behind the marker that says it is complete."""
    tmp = path + ".tmp"
    writer(tmp)
    os.replace(tmp, path)


def strip_csv_vectors(csv_path):
    """Rewrite a track-descriptor CSV with empty vector columns. The vectors
    live in the bundle's .npy from then on; uids, track references and
    history (what result assembly needs) are kept."""
    tmp = csv_path + ".tmp"
    with open(csv_path) as src, open(tmp, 'w') as dst:
        for line in src:
            stripped = line.rstrip('\n')
            if not stripped or stripped.startswith('#'):
                dst.write(line)
                continue
            parts = stripped.split(',')
            if len(parts) == 8:
                parts[4] = '0'
                parts[5] = ' '
                dst.write(','.join(parts) + '\n')
            else:
                dst.write(line)
    os.replace(tmp, csv_path)


def convert_bundle_vectors(database_dir, name, verbose=True):
    """Make sure <name>_descriptors.npy / <name>_uids.txt exist and are
    current with respect to <name>_descriptors.csv. Returns (row count,
    converted): the count is None when the video has no descriptors at all,
    and converted is True when the array was (re)written from the CSV."""
    csv_path = os.path.join(database_dir, name + BUNDLE_DESCRIPTOR_CSV_POSTFIX)
    npy_path = os.path.join(database_dir, name + BUNDLE_DESCRIPTOR_NPY_POSTFIX)
    uid_path = os.path.join(database_dir, name + BUNDLE_UIDS_POSTFIX)

    have_npy = os.path.exists(npy_path) and os.path.exists(uid_path)
    csv_newer = (os.path.exists(csv_path) and
                 (not have_npy or os.path.getmtime(csv_path) > os.path.getmtime(npy_path)))

    if have_npy and not csv_newer:
        return int(np.load(npy_path, mmap_mode='r').shape[0]), False

    if not os.path.exists(csv_path):
        return None, False

    source = KwiverCsvDescriptorSource(csv_path)
    uids, descs = source.get_descriptors()
    if len(uids) == 0:
        if have_npy:
            # The CSV was stripped after conversion; the array is the truth
            return int(np.load(npy_path, mmap_mode='r').shape[0]), False
        return None, False

    def write_array(tmp):
        # A file object keeps np.save from appending .npy to the temp name
        with open(tmp, 'wb') as f:
            np.save(f, np.asarray(descs, dtype=np.float32))
    _write_atomic(npy_path, write_array)

    def write_uids(tmp):
        with open(tmp, 'w') as f:
            for uid in uids:
                f.write(uid + '\n')
    _write_atomic(uid_path, write_uids)

    if verbose:
        print("    %s: converted %d descriptors (%d-d) to float32" % (
            name, len(uids), descs.shape[1]))
    return len(uids), True


def build_index_bundles(database_dir, bit_length=256, itq_iterations=100,
                        random_seed=0, normalize=None, pca_method='cov_eig',
                        init_method='svd', max_train_descriptors=100000,
                        retrain=False, strip_vectors=True, verbose=True):
    """Build or refresh the file-backed index in ``database_dir``.

    1. Every <name>_descriptors.csv is converted to a float32 array bundle.
    2. The shared ITQ model is trained (on a sample across all bundles)
       when absent or when ``retrain`` is set.
    3. Every bundle whose manifest does not name the current model gets
       its hash codes computed and its manifest written.

    Returns a summary dict: bundles, descriptors, rehashed, trained.
    """
    database_dir = os.path.abspath(database_dir)
    if not os.path.isdir(database_dir):
        raise ValueError("Index folder does not exist: %s" % database_dir)
    itq_dir = os.path.join(database_dir, "ITQ")
    os.makedirs(itq_dir, exist_ok=True)

    if random_seed is not None:
        np.random.seed(random_seed)

    if verbose:
        print("  (1/3) Converting descriptor files...")
    names = list_index_bundles(database_dir)
    counts = {}
    reconverted = set()
    for name in names:
        count, converted = convert_bundle_vectors(database_dir, name, verbose=verbose)
        if count:
            counts[name] = count
            if converted:
                # A re-ingested video may keep its descriptor count while its
                # vectors change, so a fresh array always gets fresh codes.
                reconverted.add(name)
    total = sum(counts.values())
    if verbose:
        print("    %d video(s), %d descriptors" % (len(counts), total))

    suffix = _model_suffix(bit_length, itq_iterations, random_seed)
    model = ITQModel(bit_length=bit_length, itq_iterations=itq_iterations,
                     random_seed=random_seed, normalize=normalize,
                     pca_method=pca_method, init_method=init_method)
    trained = False

    if verbose:
        print("  (2/3) ITQ model...")
    if not retrain and _model_present(itq_dir, suffix):
        model.load(itq_dir)
        if verbose:
            print("    Using existing model %s" % suffix)
    else:
        if total == 0:
            raise ValueError("No descriptors found in %s to train the ITQ model on"
                             % database_dir)
        # Sample training rows proportionally across bundles
        sample = []
        budget = max_train_descriptors if max_train_descriptors else total
        for name, count in counts.items():
            arr = np.load(os.path.join(database_dir, name + BUNDLE_DESCRIPTOR_NPY_POSTFIX),
                          mmap_mode='r')
            take = max(1, int(round(budget * count / float(total)))) if budget < total else count
            take = min(take, count)
            rows = (np.sort(np.random.choice(count, take, replace=False))
                    if take < count else np.arange(count))
            sample.append(np.asarray(arr[rows], dtype=np.float64))
        train = np.concatenate(sample, axis=0)
        if verbose:
            print("    Training on %d descriptors (%d-d)" % (train.shape[0], train.shape[1]))
        model.fit(train, verbose=verbose)
        model.save(itq_dir)
        trained = True
        if verbose:
            print("    Saved model %s to %s" % (suffix, itq_dir))

    model_hash = _model_hash(itq_dir, suffix)

    if verbose:
        print("  (3/3) Hash codes...")
    rehashed = 0
    for name, count in counts.items():
        manifest = read_bundle_manifest(database_dir, name)
        hashes_path = os.path.join(database_dir, name + BUNDLE_HASHES_POSTFIX)
        current = (manifest is not None
                   and name not in reconverted
                   and manifest.get("itq_model") == suffix
                   and manifest.get("itq_model_hash") == model_hash
                   and manifest.get("count") == count
                   and os.path.exists(hashes_path))
        if current:
            continue
        arr = np.load(os.path.join(database_dir, name + BUNDLE_DESCRIPTOR_NPY_POSTFIX),
                      mmap_mode='r')
        codes = []
        batch = 10000
        for start in range(0, count, batch):
            chunk = np.asarray(arr[start:start + batch], dtype=np.float64)
            codes.append(model.compute_hashes_bool(chunk).astype(np.uint8))
        codes = np.concatenate(codes, axis=0) if codes else np.zeros((0, bit_length), np.uint8)
        def write_codes(tmp, codes=codes):
            with open(tmp, 'wb') as f:
                np.save(f, codes)
        _write_atomic(hashes_path, write_codes)

        manifest = {
            "version": BUNDLE_MANIFEST_VERSION,
            "name": name,
            "count": int(count),
            "dimension": int(arr.shape[1]),
            "dtype": str(arr.dtype),
            "itq_model": suffix,
            "itq_model_hash": model_hash,
            "bit_length": int(bit_length),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(database_dir, name + BUNDLE_INDEX_POSTFIX) + ".tmp", 'w') as f:
            json.dump(manifest, f, indent=2)
        os.replace(os.path.join(database_dir, name + BUNDLE_INDEX_POSTFIX) + ".tmp",
                   os.path.join(database_dir, name + BUNDLE_INDEX_POSTFIX))
        rehashed += 1
        if verbose:
            print("    %s: %d hash codes" % (name, count))

        if strip_vectors:
            csv_path = os.path.join(database_dir, name + BUNDLE_DESCRIPTOR_CSV_POSTFIX)
            if os.path.exists(csv_path):
                strip_csv_vectors(csv_path)

    return {"bundles": len(counts), "descriptors": total,
            "rehashed": rehashed, "trained": trained}


def remove_index_bundle(database_dir, name):
    """Delete every file of one indexed video. Returns the removed paths."""
    removed = []
    for postfix in (BUNDLE_INDEX_POSTFIX, BUNDLE_DESCRIPTOR_CSV_POSTFIX,
                    BUNDLE_DESCRIPTOR_NPY_POSTFIX, BUNDLE_UIDS_POSTFIX,
                    BUNDLE_HASHES_POSTFIX, "_tracks.csv"):
        path = os.path.join(database_dir, name + postfix)
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    return removed


class PostgresDescriptorSource(DescriptorSource):
    """Load descriptors from PostgreSQL database."""

    def __init__(self, host="localhost", port=5432, dbname="postgres",
                 user="postgres", password=None, table_name="DESCRIPTOR",
                 uuid_col="UID", element_col="VECTOR_DATA"):
        """
        Initialize PostgreSQL descriptor source.

        Args:
            host: Database host
            port: Database port
            dbname: Database name
            user: Database user
            password: Database password (optional)
            table_name: Table containing descriptors
            uuid_col: Column name for UIDs (default: "uid")
            element_col: Column name for descriptor data (default: "element")
        """
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.table_name = table_name
        self.uuid_col = uuid_col
        self.element_col = element_col
        self._conn = None
        self._count = None

    def _connect(self):
        """Establish database connection."""
        if self._conn is not None:
            return self._conn

        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support. "
                "Install with: pip install psycopg2-binary"
            )

        conn_params = {
            'host': self.host,
            'port': self.port,
            'dbname': self.dbname,
            'user': self.user,
        }
        if self.password:
            conn_params['password'] = self.password

        self._conn = psycopg2.connect(**conn_params)
        return self._conn

    def get_descriptors(self, max_count=None, uids=None, random_sample=False):
        conn = self._connect()
        cursor = conn.cursor()

        if uids is not None:
            # Fetch specific UIDs
            placeholders = ','.join(['%s'] * len(uids))
            query = f"SELECT {self.uuid_col}, {self.element_col} FROM {self.table_name} WHERE {self.uuid_col} IN ({placeholders})"
            cursor.execute(query, uids)
        elif max_count is not None:
            if random_sample:
                # Random sampling using ORDER BY RANDOM()
                query = f"SELECT {self.uuid_col}, {self.element_col} FROM {self.table_name} ORDER BY RANDOM() LIMIT %s"
            else:
                query = f"SELECT {self.uuid_col}, {self.element_col} FROM {self.table_name} LIMIT %s"
            cursor.execute(query, (max_count,))
        else:
            query = f"SELECT {self.uuid_col}, {self.element_col} FROM {self.table_name}"
            cursor.execute(query)

        uids_list = []
        descriptors = []

        for row in cursor:
            uid, element = row
            # element is typically a pickled numpy array or list
            if isinstance(element, (bytes, memoryview)):
                try:
                    values = pickle.loads(bytes(element))
                    if hasattr(values, 'tolist'):
                        values = values.tolist()
                except:
                    continue
            elif isinstance(element, str):
                # CSV or PostgreSQL array format
                try:
                    # Remove curly braces if present (PostgreSQL array format)
                    element = element.strip()
                    if element.startswith('{') and element.endswith('}'):
                        element = element[1:-1]
                    values = [float(x) for x in element.split(',') if x.strip()]
                except ValueError:
                    continue
            elif isinstance(element, (list, tuple)):
                # PostgreSQL array already parsed by psycopg2
                values = list(element)
            else:
                values = element

            if values:
                uids_list.append(uid)
                descriptors.append(values)

        cursor.close()
        return uids_list, np.array(descriptors) if descriptors else np.array([])

    def get_all_uids(self):
        conn = self._connect()
        cursor = conn.cursor()

        query = f"SELECT {self.uuid_col} FROM {self.table_name}"
        cursor.execute(query)

        uids = [row[0] for row in cursor]
        cursor.close()
        return uids

    def __len__(self):
        if self._count is not None:
            return self._count

        conn = self._connect()
        cursor = conn.cursor()

        query = f"SELECT COUNT(*) FROM {self.table_name}"
        cursor.execute(query)
        self._count = cursor.fetchone()[0]
        cursor.close()
        return self._count


class Hash2UUIDStore:
    """Store mapping from hash codes to descriptor UIDs."""

    def __init__(self):
        self._store = defaultdict(list)

    def add(self, hash_code, uid):
        """Add a hash -> UID mapping."""
        self._store[hash_code].append(uid)

    def add_batch(self, hash_codes, uids):
        """Add multiple hash -> UID mappings."""
        for h, u in zip(hash_codes, uids):
            self._store[h].append(u)

    def get(self, hash_code):
        """Get UIDs for a hash code."""
        return self._store.get(hash_code, [])

    def get_all_uids(self):
        """Get all UIDs currently in the store."""
        all_uids = set()
        for uids in self._store.values():
            all_uids.update(uids)
        return all_uids

    def save(self, file_path):
        """Save store to pickle file."""
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(dict(self._store), f, protocol=-1)

    def load(self, file_path):
        """Load store from pickle file."""
        with open(file_path, 'rb') as f:
            self._store = defaultdict(list, pickle.load(f))
        return self

    def __len__(self):
        return len(self._store)


class LinearHashIndex:
    """
    Linear hash index for nearest neighbor search.

    This class stores unique hash codes as integers for efficient
    hamming distance computation.
    """

    def __init__(self, bit_length=256):
        """
        Initialize linear hash index.

        Args:
            bit_length: Number of bits in hash codes (default: 256)
        """
        self.bit_length = bit_length
        self.index = set()

    def _bytes_to_int(self, hash_bytes):
        """Convert hash bytes to integer."""
        return int.from_bytes(hash_bytes, byteorder='big')

    def _int_to_bytes(self, hash_int):
        """Convert integer to hash bytes."""
        n_bytes = (self.bit_length + 7) // 8
        return hash_int.to_bytes(n_bytes, byteorder='big')

    def add(self, hash_code):
        """Add a hash code to the index."""
        if isinstance(hash_code, bytes):
            hash_int = self._bytes_to_int(hash_code)
        else:
            hash_int = hash_code
        self.index.add(hash_int)

    def add_batch(self, hash_codes):
        """Add multiple hash codes to the index."""
        for h in hash_codes:
            self.add(h)

    def build_from_hash2uuid(self, hash2uuid_store):
        """Build index from Hash2UUIDStore keys."""
        for hash_code in hash2uuid_store._store.keys():
            self.add(hash_code)

    def save(self, file_path):
        """
        Save index to numpy file.
        """
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        # Save as array of integers
        np.save(file_path, np.array(list(self.index), dtype=object))

    def load(self, file_path):
        """Load index from numpy file."""
        data = np.load(file_path, allow_pickle=True)
        self.index = set(data)
        return self

    def hamming_distance(self, a, b):
        """Compute hamming distance between two integers."""
        xor = a ^ b
        count = 0
        while xor:
            count += xor & 1
            xor >>= 1
        return count

    def nearest_neighbors(self, query_hash, n=1):
        """
        Find n nearest neighbors by hamming distance.

        Args:
            query_hash: Query hash code (bytes or int)
            n: Number of neighbors to return

        Returns:
            List of (hash_int, normalized_distance) tuples
        """
        if isinstance(query_hash, bytes):
            query_int = self._bytes_to_int(query_hash)
        else:
            query_int = query_hash

        # Compute distances to all indexed hashes
        distances = []
        for h in self.index:
            dist = self.hamming_distance(query_int, h)
            distances.append((h, dist / self.bit_length))

        # Sort by distance and return top n
        distances.sort(key=lambda x: x[1])
        return distances[:n]

    def count(self):
        """Return number of indexed hash codes."""
        return len(self.index)

    def __len__(self):
        return len(self.index)


def load_uuids_list(filepath):
    """
    Load list of UUIDs from a file (one UUID per line).

    Args:
        filepath: Path to file containing UUIDs

    Returns:
        List of UUID strings
    """
    uids = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                uids.append(line)
    return uids


def generate_nn_index(descriptor_source, output_dir, bit_length=256,
                      itq_iterations=100, random_seed=0, normalize=None,
                      pca_method='cov_eig', init_method='svd',
                      max_train_descriptors=100000, random_sample=True,
                      train_uids=None, report_interval=1.0,
                      incremental=False, verbose=True):
    """
    Generate ITQ LSH index from descriptors for nearest neighbor search.

    This is the main entry point for building an ITQ-based LSH index.

    Args:
        descriptor_source: DescriptorSource instance
        output_dir: Directory to store ITQ model and hash mappings
        bit_length: Number of bits in hash code (default: 256)
        itq_iterations: Number of ITQ refinement iterations (default: 100)
        random_seed: Random seed for reproducibility (default: 0)
        normalize: Normalization order (default: None = no normalization)
        pca_method: PCA method ('cov_eig' or 'direct_svd', default: 'cov_eig')
        init_method: Rotation init method ('svd' or 'qr', default: 'svd')
        max_train_descriptors: Max descriptors for training (default: 100000)
        random_sample: Randomly sample training descriptors if max < total (default: True)
        train_uids: Optional list of specific UIDs to use for training (default: None)
        report_interval: Seconds between progress reports (default: 1.0)
        incremental: If True, only compute hashes for new descriptors (default: False)
        verbose: Print progress messages (default: True)

    Returns:
        Tuple of (model, hash2uuid_store, linear_hash_index)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed for reproducible subsampling
    if random_seed is not None:
        np.random.seed(random_seed)

    # Check for existing hash2uuid store for incremental updates
    hash2uuid_path = os.path.join(output_dir, "hash2uuids.memKvStore.pickle")
    existing_uids = set()
    hash2uuid = Hash2UUIDStore()

    if incremental and os.path.exists(hash2uuid_path):
        if verbose:
            print("  Loading existing hash2uuid store for incremental update...")
        hash2uuid.load(hash2uuid_path)
        existing_uids = hash2uuid.get_all_uids()
        if verbose:
            print(f"    Found {len(existing_uids)} existing UIDs")

    # Step 1: Train ITQ model
    if verbose:
        print("  (1/3) Training ITQ Model...")

    if train_uids is not None:
        # Use specific UIDs for training
        if verbose:
            print(f"    Using {len(train_uids)} specified UIDs for training")
        if max_train_descriptors and max_train_descriptors < len(train_uids):
            if random_sample:
                train_uids = list(np.random.choice(train_uids, max_train_descriptors, replace=False))
            else:
                train_uids = train_uids[:max_train_descriptors]
        train_uids_list, train_descs = descriptor_source.get_descriptors(uids=train_uids)
    else:
        train_uids_list, train_descs = descriptor_source.get_descriptors(
            max_count=max_train_descriptors,
            random_sample=random_sample
        )

    if len(train_descs) == 0:
        raise ValueError("No descriptors found for training")

    if verbose:
        total_available = len(descriptor_source) if hasattr(descriptor_source, '__len__') else "unknown"
        print(f"    Training on {len(train_descs)} descriptors (total available: {total_available})")

    model = ITQModel(
        bit_length=bit_length,
        itq_iterations=itq_iterations,
        random_seed=random_seed,
        normalize=normalize,
        pca_method=pca_method,
        init_method=init_method
    )
    model.fit(train_descs, verbose=verbose, report_interval=report_interval)

    # Save model
    mean_path, rotation_path = model.save(output_dir)
    if verbose:
        print(f"    Saved mean vector to: {mean_path}")
        print(f"    Saved rotation matrix to: {rotation_path}")
        print("  Success")

    # Step 2: Compute hash codes for all descriptors
    if verbose:
        print("  (2/3) Computing Hash Codes...")

    # Get all descriptors (may be more than training set)
    all_uids, all_descs = descriptor_source.get_descriptors()

    # Filter out already-processed UIDs for incremental updates
    if incremental and existing_uids:
        new_indices = [i for i, uid in enumerate(all_uids) if uid not in existing_uids]
        if verbose:
            print(f"    Skipping {len(all_uids) - len(new_indices)} already-processed descriptors")
        all_uids = [all_uids[i] for i in new_indices]
        all_descs = all_descs[new_indices] if len(new_indices) > 0 else np.array([])

    if len(all_descs) > 0:
        if verbose:
            print(f"    Computing hashes for {len(all_uids)} descriptors...")

        # Compute hashes in batches for memory efficiency
        batch_size = 10000
        total_hashed = 0

        for i in range(0, len(all_descs), batch_size):
            batch_descs = all_descs[i:i + batch_size]
            batch_uids = all_uids[i:i + batch_size]

            hash_codes = model.compute_hashes(batch_descs)
            hash2uuid.add_batch(hash_codes, batch_uids)

            total_hashed += len(batch_descs)
            if verbose and total_hashed % 50000 == 0:
                print(f"      Processed {total_hashed}/{len(all_descs)} descriptors...")
    else:
        if verbose:
            print("    No new descriptors to process")

    # Save hash -> UUID mapping
    hash2uuid.save(hash2uuid_path)

    if verbose:
        print(f"    Saved hash2uuid mapping to: {hash2uuid_path}")
        print(f"    Total unique hash codes: {len(hash2uuid)}")
        print("  Success")

    # Save C++-friendly format: hash codes as boolean array + UIDs as text
    if verbose:
        print("  Saving C++-friendly hash code format...")

    # Reload all descriptors to compute boolean hashes in order
    all_uids_ordered, all_descs_ordered = descriptor_source.get_descriptors()
    if len(all_descs_ordered) > 0:
        hash_codes_bool = model.compute_hashes_bool(all_descs_ordered)

        # Save hash codes as uint8 (0/1) array - easy to read in C++
        hash_codes_path = os.path.join(output_dir, "lsh_hash_codes.npy")
        np.save(hash_codes_path, hash_codes_bool.astype(np.uint8))

        # Save UIDs as text file (one per line, same order as hash codes)
        hash_uids_path = os.path.join(output_dir, "lsh_hash_uids.txt")
        with open(hash_uids_path, 'w') as f:
            for uid in all_uids_ordered:
                f.write(uid + '\n')

        if verbose:
            print(f"    Saved hash codes to: {hash_codes_path}")
            print(f"    Saved UIDs to: {hash_uids_path}")
            print(f"    Total: {len(all_uids_ordered)} descriptors")

    # Step 3: Build LinearHashIndex
    if verbose:
        print("  (3/3) Building Linear Hash Index...")

    linear_index = LinearHashIndex(bit_length=bit_length)
    linear_index.build_from_hash2uuid(hash2uuid)

    linear_index_path = os.path.join(output_dir, "linearhashindex.npy")
    linear_index.save(linear_index_path)

    if verbose:
        print(f"    Saved linear hash index to: {linear_index_path}")
        print(f"    Indexed {len(linear_index)} unique hash codes")
        print("  Success")

    return model, hash2uuid, linear_index


def compute_hashes_only(descriptor_source, model_dir, output_dir=None,
                        bit_length=256, itq_iterations=100, random_seed=0,
                        normalize=None, incremental=False, verbose=True):
    """
    Compute hash codes using an existing trained model.

    Args:
        descriptor_source: DescriptorSource instance
        model_dir: Directory containing trained model files
        output_dir: Directory to store hash mappings (default: model_dir)
        bit_length: Number of bits in hash code (must match trained model)
        itq_iterations: Number of ITQ iterations (must match trained model)
        random_seed: Random seed (must match trained model)
        normalize: Normalization order (must match trained model)
        incremental: If True, only compute hashes for new descriptors (default: False)
        verbose: Print progress messages (default: True)

    Returns:
        Tuple of (hash2uuid_store, linear_hash_index)
    """
    if output_dir is None:
        output_dir = model_dir

    if verbose:
        print("  Loading existing ITQ model...")

    model = ITQModel(
        bit_length=bit_length,
        itq_iterations=itq_iterations,
        random_seed=random_seed,
        normalize=normalize
    )
    model.load(model_dir)

    if verbose:
        print(f"    Loaded model from: {model_dir}")

    # Check for existing hash2uuid store for incremental updates
    hash2uuid_path = os.path.join(output_dir, "hash2uuids.memKvStore.pickle")
    existing_uids = set()
    hash2uuid = Hash2UUIDStore()

    if incremental and os.path.exists(hash2uuid_path):
        if verbose:
            print("  Loading existing hash2uuid store for incremental update...")
        hash2uuid.load(hash2uuid_path)
        existing_uids = hash2uuid.get_all_uids()
        if verbose:
            print(f"    Found {len(existing_uids)} existing UIDs")

    if verbose:
        print("  (1/2) Computing Hash Codes...")

    # Get all descriptors
    all_uids, all_descs = descriptor_source.get_descriptors()

    # Filter out already-processed UIDs for incremental updates
    if incremental and existing_uids:
        new_indices = [i for i, uid in enumerate(all_uids) if uid not in existing_uids]
        if verbose:
            print(f"    Skipping {len(all_uids) - len(new_indices)} already-processed descriptors")
        all_uids = [all_uids[i] for i in new_indices]
        all_descs = all_descs[new_indices] if len(new_indices) > 0 else np.array([])

    if len(all_descs) > 0:
        if verbose:
            print(f"    Computing hashes for {len(all_uids)} descriptors...")

        # Compute hashes in batches
        batch_size = 10000
        total_hashed = 0

        for i in range(0, len(all_descs), batch_size):
            batch_descs = all_descs[i:i + batch_size]
            batch_uids = all_uids[i:i + batch_size]

            hash_codes = model.compute_hashes(batch_descs)
            hash2uuid.add_batch(hash_codes, batch_uids)

            total_hashed += len(batch_descs)
            if verbose and total_hashed % 50000 == 0:
                print(f"      Processed {total_hashed}/{len(all_descs)} descriptors...")
    else:
        if verbose:
            print("    No new descriptors to process")

    # Save hash -> UUID mapping
    hash2uuid.save(hash2uuid_path)

    if verbose:
        print(f"    Saved hash2uuid mapping to: {hash2uuid_path}")
        print(f"    Total unique hash codes: {len(hash2uuid)}")
        print("  Success")

    # Save C++-friendly format: hash codes as boolean array + UIDs as text
    if verbose:
        print("  Saving C++-friendly hash code format...")

    # Reload all descriptors to compute boolean hashes in order
    all_uids_ordered, all_descs_ordered = descriptor_source.get_descriptors()
    if len(all_descs_ordered) > 0:
        hash_codes_bool = model.compute_hashes_bool(all_descs_ordered)

        # Save hash codes as uint8 (0/1) array - easy to read in C++
        hash_codes_path = os.path.join(output_dir, "lsh_hash_codes.npy")
        np.save(hash_codes_path, hash_codes_bool.astype(np.uint8))

        # Save UIDs as text file (one per line, same order as hash codes)
        hash_uids_path = os.path.join(output_dir, "lsh_hash_uids.txt")
        with open(hash_uids_path, 'w') as f:
            for uid in all_uids_ordered:
                f.write(uid + '\n')

        if verbose:
            print(f"    Saved hash codes to: {hash_codes_path}")
            print(f"    Saved UIDs to: {hash_uids_path}")
            print(f"    Total: {len(all_uids_ordered)} descriptors")

    # Build LinearHashIndex
    if verbose:
        print("  (2/2) Building Linear Hash Index...")

    linear_index = LinearHashIndex(bit_length=bit_length)
    linear_index.build_from_hash2uuid(hash2uuid)

    linear_index_path = os.path.join(output_dir, "linearhashindex.npy")
    linear_index.save(linear_index_path)

    if verbose:
        print(f"    Saved linear hash index to: {linear_index_path}")
        print(f"    Indexed {len(linear_index)} unique hash codes")
        print("  Success")

    return hash2uuid, linear_index


def load_config(config_path):
    """
    Load configuration from JSON file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract relevant parameters
    result = {}

    # ITQ parameters
    if 'itq_config' in config:
        itq = config['itq_config']
        result['bit_length'] = itq.get('bit_length', 256)
        result['itq_iterations'] = itq.get('itq_iterations', 100)
        result['random_seed'] = itq.get('random_seed', 0)
        result['normalize'] = itq.get('normalize')
    elif 'plugins' in config and 'lsh_functor' in config['plugins']:
        lsh = config['plugins']['lsh_functor']
        if 'ItqFunctor' in lsh:
            itq = lsh['ItqFunctor']
            result['bit_length'] = itq.get('bit_length', 256)
            result['itq_iterations'] = itq.get('itq_iterations', 100)
            result['random_seed'] = itq.get('random_seed', 0)
            result['normalize'] = itq.get('normalize')

    # Descriptor source
    if 'descriptor_index' in config:
        di = config['descriptor_index']
        if di.get('type') == 'PostgresDescriptorIndex':
            pg = di.get('PostgresDescriptorIndex', {})
            result['source_type'] = 'postgres'
            result['db_host'] = pg.get('db_host', 'localhost')
            result['db_port'] = pg.get('db_port', 5432)
            result['db_name'] = pg.get('db_name', 'postgres')
            result['db_user'] = pg.get('db_user', 'postgres')
            result['db_pass'] = pg.get('db_pass')
            result['table_name'] = pg.get('table_name', 'DESCRIPTOR')
            result['uuid_col'] = pg.get('uuid_col', 'UID')
            result['element_col'] = pg.get('element_col', 'VECTOR_DATA')
    elif 'plugins' in config and 'descriptor_index' in config['plugins']:
        di = config['plugins']['descriptor_index']
        if di.get('type') == 'PostgresDescriptorIndex':
            pg = di.get('PostgresDescriptorIndex', {})
            result['source_type'] = 'postgres'
            result['db_host'] = pg.get('db_host', 'localhost')
            result['db_port'] = pg.get('db_port', 5432)
            result['db_name'] = pg.get('db_name', 'postgres')
            result['db_user'] = pg.get('db_user', 'postgres')
            result['db_pass'] = pg.get('db_pass')
            result['table_name'] = pg.get('table_name', 'DESCRIPTOR')
            result['uuid_col'] = pg.get('uuid_col', 'UID')
            result['element_col'] = pg.get('element_col', 'VECTOR_DATA')

    # Max descriptors for training
    result['max_descriptors'] = config.get('max_descriptors', 100000)

    # UUID list file
    result['uuids_list_filepath'] = config.get('uuids_list_filepath')

    return result
