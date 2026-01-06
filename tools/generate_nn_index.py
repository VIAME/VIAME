#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Build ITQ (Iterative Quantization) LSH index for efficient nearest neighbor search.

This tool implements ITQ training and hash code computation using only numpy
and standard Python libraries.

The ITQ algorithm:
1. Optionally normalizes input descriptors
2. Centers the data by subtracting the mean
3. Projects data to bit_length dimensions using PCA
4. Iteratively refines a rotation matrix to minimize quantization error
5. Produces binary hash codes for efficient similarity search

Usage:
    # Train and compute hashes from CSV file
    python generate_nn_index.py --descriptor-file descriptors.csv --output-dir database/ITQ

    # Compute hashes only using existing model
    python generate_nn_index.py --descriptor-file descriptors.csv --model-dir database/ITQ --hash-only

References:
    Gong, Y., & Lazebnik, S. (2011). Iterative quantization: A procrustean approach
    to learning binary codes. In CVPR.
    http://www.cs.unc.edu/~lazebnik/publications/cvpr11_small_code.pdf
"""

import argparse
import json
import os
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
                # CSV format
                try:
                    values = [float(x) for x in element.split(',')]
                except ValueError:
                    continue
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


def main():
    parser = argparse.ArgumentParser(
        description='Build ITQ LSH index for efficient nearest neighbor search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--config', '-c',
        help='Path to JSON config file'
    )
    input_group.add_argument(
        '--descriptor-file', '-f',
        help='Path to CSV file containing descriptors (format: uid,val1,val2,...)'
    )

    # Database options (when not using config)
    parser.add_argument(
        '--db-host',
        default='localhost',
        help='PostgreSQL host (default: localhost)'
    )
    parser.add_argument(
        '--db-port',
        type=int,
        default=5432,
        help='PostgreSQL port (default: 5432)'
    )
    parser.add_argument(
        '--db-name',
        default='postgres',
        help='PostgreSQL database name (default: postgres)'
    )
    parser.add_argument(
        '--db-user',
        default='postgres',
        help='PostgreSQL user (default: postgres)'
    )
    parser.add_argument(
        '--db-pass',
        default=None,
        help='PostgreSQL password'
    )
    parser.add_argument(
        '--table-name',
        default='DESCRIPTOR',
        help='Table name containing descriptors (default: DESCRIPTOR)'
    )
    parser.add_argument(
        '--uuid-col',
        default='UID',
        help='Column name for descriptor UIDs (default: UID)'
    )
    parser.add_argument(
        '--element-col',
        default='VECTOR_DATA',
        help='Column name for descriptor data (default: VECTOR_DATA)'
    )

    # ITQ parameters
    parser.add_argument(
        '--bit-length', '-b',
        type=int,
        default=256,
        help='Number of bits in hash code (default: 256)'
    )
    parser.add_argument(
        '--itq-iterations', '-i',
        type=int,
        default=100,
        help='Number of ITQ refinement iterations (default: 100)'
    )
    parser.add_argument(
        '--random-seed', '-r',
        type=int,
        default=0,
        help='Random seed for reproducibility (default: 0)'
    )
    parser.add_argument(
        '--normalize',
        type=float,
        default=None,
        help='Normalization order for input vectors (default: None = no normalization). '
             'Use 2 for L2 normalization.'
    )
    parser.add_argument(
        '--max-train-descriptors', '-m',
        type=int,
        default=100000,
        help='Maximum descriptors for training (default: 100000)'
    )

    # Training options
    parser.add_argument(
        '--uuids-list',
        help='Path to file containing UIDs to use for training (one per line).'
    )
    parser.add_argument(
        '--no-random-sample',
        action='store_true',
        help='Disable random subsampling when max_train_descriptors < total. '
             'By default, descriptors are randomly sampled.'
    )

    # Algorithm options
    parser.add_argument(
        '--pca-method',
        choices=['cov_eig', 'direct_svd'],
        default='cov_eig',
        help='PCA computation method (default: cov_eig). '
             'cov_eig: covariance + eigendecomposition, '
             'direct_svd: direct SVD (more numerically stable)'
    )
    parser.add_argument(
        '--init-method',
        choices=['svd', 'qr'],
        default='svd',
        help='Rotation initialization method (default: svd). '
             'svd: SVD orthogonalization, qr: QR decomposition'
    )

    # Hash-only mode
    parser.add_argument(
        '--hash-only',
        action='store_true',
        help='Only compute hashes using existing model (skip training)'
    )
    parser.add_argument(
        '--model-dir',
        help='Directory containing existing model (required for --hash-only)'
    )

    # Incremental mode
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Only process new descriptors (skip already-hashed UIDs). '
             'Loads existing hash2uuid store and adds only new entries.'
    )

    # Output
    parser.add_argument(
        '--output-dir', '-o',
        default='database/ITQ',
        help='Output directory for model and hash mappings (default: database/ITQ)'
    )

    # Progress reporting
    parser.add_argument(
        '--report-interval',
        type=float,
        default=1.0,
        help='Seconds between progress reports (default: 1.0)'
    )

    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Print progress messages (default: True)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    # Validate hash-only mode
    if args.hash_only and not args.model_dir:
        parser.error("--model-dir is required when using --hash-only")

    try:
        # Load configuration
        if args.config:
            if verbose:
                print(f"Loading configuration from: {args.config}")
            config = load_config(args.config)

            bit_length = config.get('bit_length', args.bit_length)
            itq_iterations = config.get('itq_iterations', args.itq_iterations)
            random_seed = config.get('random_seed', args.random_seed)
            normalize = config.get('normalize', args.normalize)
            max_train = config.get('max_descriptors', args.max_train_descriptors)
            uuids_list_filepath = config.get('uuids_list_filepath') or args.uuids_list

            if config.get('source_type') == 'postgres':
                source = PostgresDescriptorSource(
                    host=config.get('db_host', 'localhost'),
                    port=config.get('db_port', 5432),
                    dbname=config.get('db_name', 'postgres'),
                    user=config.get('db_user', 'postgres'),
                    password=config.get('db_pass'),
                    table_name=config.get('table_name', 'descriptor_index'),
                    uuid_col=config.get('uuid_col', 'uid'),
                    element_col=config.get('element_col', 'element')
                )
            else:
                raise ValueError("Config does not specify a valid descriptor source")
        elif args.descriptor_file:
            source = CSVDescriptorSource(args.descriptor_file)
            bit_length = args.bit_length
            itq_iterations = args.itq_iterations
            random_seed = args.random_seed
            normalize = args.normalize
            max_train = args.max_train_descriptors
            uuids_list_filepath = args.uuids_list
        else:
            # Use database with command-line args
            source = PostgresDescriptorSource(
                host=args.db_host,
                port=args.db_port,
                dbname=args.db_name,
                user=args.db_user,
                password=args.db_pass,
                table_name=args.table_name,
                uuid_col=args.uuid_col,
                element_col=args.element_col
            )
            bit_length = args.bit_length
            itq_iterations = args.itq_iterations
            random_seed = args.random_seed
            normalize = args.normalize
            max_train = args.max_train_descriptors
            uuids_list_filepath = args.uuids_list

        # Load UIDs list if specified
        train_uids = None
        if uuids_list_filepath and os.path.isfile(uuids_list_filepath):
            if verbose:
                print(f"Loading UIDs list from: {uuids_list_filepath}")
            train_uids = load_uuids_list(uuids_list_filepath)
            if verbose:
                print(f"  Loaded {len(train_uids)} UIDs")

        if args.hash_only:
            # Hash-only mode
            hash2uuid, linear_index = compute_hashes_only(
                descriptor_source=source,
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                bit_length=bit_length,
                itq_iterations=itq_iterations,
                random_seed=random_seed,
                normalize=normalize,
                incremental=args.incremental,
                verbose=verbose
            )
        else:
            # Full train + hash mode
            model, hash2uuid, linear_index = generate_nn_index(
                descriptor_source=source,
                output_dir=args.output_dir,
                bit_length=bit_length,
                itq_iterations=itq_iterations,
                random_seed=random_seed,
                normalize=normalize,
                pca_method=args.pca_method,
                init_method=args.init_method,
                max_train_descriptors=max_train,
                random_sample=not args.no_random_sample,
                train_uids=train_uids,
                report_interval=args.report_interval,
                incremental=args.incremental,
                verbose=verbose
            )

        if verbose:
            print("\nITQ index build complete!")
            if not args.hash_only:
                print(f"  Model files: {args.output_dir}/itq.model.*.npy")
            print(f"  Hash mapping: {args.output_dir}/hash2uuids.memKvStore.pickle")
            print(f"  Hash index: {args.output_dir}/linearhashindex.npy")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
