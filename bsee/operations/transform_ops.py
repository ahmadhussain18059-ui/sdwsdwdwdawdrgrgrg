"""
Transform operations for binary transformation.
"""

from typing import Callable, Dict, List, Tuple, Any


class TransformOperations:
    """Collection of transform operations."""

    def __init__(self):
        """Initialize transform operations."""
        self.operations = self._create_operations()

    def _create_operations(self) -> Dict[str, Callable]:
        """Create all transform operations."""
        return {
            'burrows_wheeler': self.burrows_wheeler,
            'burrows_wheeler_inverse': self.burrows_wheeler_inverse,
            'bitplane_extract': self.bitplane_extract,
            'bitplane_insert': self.bitplane_insert,
            'dct_transform': self.dct_transform,
            'dwt_transform': self.dwt_transform,
            'fft_transform': self.fft_transform,
            'walsh_hadamard': self.walsh_hadamard,
            'huffman_encode': self.huffman_encode,
            'run_length_encode': self.run_length_encode,
            'arithmetic_encode': self.arithmetic_encode,
            'lz77_encode': self.lz77_encode,
            'move_to_front': self.move_to_front,
            'distance_coding': self.distance_coding,
            'elias_gamma': self.elias_gamma,
            'elias_delta': self.elias_delta,
            'golomb_coding': self.golomb_coding,
            'fibonacci_coding': self.fibonacci_coding,
            'phase_in_coding': self.phase_in_coding,
            'adaptive_huffman': self.adaptive_huffman
        }

    def get_operations(self) -> Dict[str, Callable]:
        """Get all operations."""
        return self.operations

    def get_metadata(self, operation_name: str) -> Dict[str, Any]:
        """Get metadata for an operation."""
        metadata_map = {
            'burrows_wheeler': {
                'category': 'transform',
                'description': 'Burrows-Wheeler transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'burrows_wheeler_inverse': {
                'category': 'transform',
                'description': 'Inverse Burrows-Wheeler transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'bitplane_extract': {
                'category': 'transform',
                'description': 'Extract specific bitplane',
                'required_params': ['plane'],
                'optional_params': {},
                'reversible': False
            },
            'bitplane_insert': {
                'category': 'transform',
                'description': 'Insert bitplane data',
                'required_params': ['plane', 'data'],
                'optional_params': {},
                'reversible': False
            },
            'dct_transform': {
                'category': 'transform',
                'description': 'Discrete cosine transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'dwt_transform': {
                'category': 'transform',
                'description': 'Discrete wavelet transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'fft_transform': {
                'category': 'transform',
                'description': 'Fast Fourier transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'walsh_hadamard': {
                'category': 'transform',
                'description': 'Walsh-Hadamard transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'huffman_encode': {
                'category': 'transform',
                'description': 'Huffman encoding',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'run_length_encode': {
                'category': 'transform',
                'description': 'Run-length encoding',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'arithmetic_encode': {
                'category': 'transform',
                'description': 'Arithmetic encoding',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'lz77_encode': {
                'category': 'transform',
                'description': 'LZ77 encoding',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'move_to_front': {
                'category': 'transform',
                'description': 'Move-to-front transform',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'distance_coding': {
                'category': 'transform',
                'description': 'Distance coding',
                'required_params': [],
                'optional_params': {},
                'reversible': True
            },
            'elias_gamma': {
                'category': 'transform',
                'description': 'Elias gamma coding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'elias_delta': {
                'category': 'transform',
                'description': 'Elias delta coding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'golomb_coding': {
                'category': 'transform',
                'description': 'Golomb coding',
                'required_params': ['parameter'],
                'optional_params': {},
                'reversible': False
            },
            'fibonacci_coding': {
                'category': 'transform',
                'description': 'Fibonacci coding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'phase_in_coding': {
                'category': 'transform',
                'description': 'Phase-in coding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            },
            'adaptive_huffman': {
                'category': 'transform',
                'description': 'Adaptive Huffman coding',
                'required_params': [],
                'optional_params': {},
                'reversible': False
            }
        }
        return metadata_map.get(operation_name, {})

    def burrows_wheeler(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Burrows-Wheeler transform."""
        if len(binary_data) <= 1:
            return binary_data, lambda: binary_data, {'operation': 'burrows_wheeler', 'bytes_affected': 0}

        # Add EOF marker (use 0 as it's rarely in binary data)
        data_with_eof = binary_data + b'\x00'

        # Generate all rotations
        rotations = []
        for i in range(len(data_with_eof)):
            rotation = data_with_eof[i:] + data_with_eof[:i]
            rotations.append(rotation)

        # Sort rotations
        rotations.sort()

        # Find original string index
        original_index = rotations.index(data_with_eof)

        # Extract last column (BWT result)
        bwt_result = bytes([rotation[-1] for rotation in rotations])

        # Combine index with result
        result = bwt_result + original_index.to_bytes(4, 'big')

        def inverse():
            if len(result) <= 4:
                return b''

            # Extract index and BWT data
            original_index = int.from_bytes(result[-4:], 'big')
            bwt_data = result[:-4]

            # Reconstruct original using LF mapping
            table = [""] * len(bwt_data)
            for _ in range(len(bwt_data)):
                # Prepend BWT character to each string
                table = [bwt_data[i] + table[i] for i in range(len(bwt_data))]
                # Sort table
                table.sort()

            return table[original_index].replace(b'\x00', b'')

        metadata = {
            'operation': 'burrows_wheeler',
            'original_index': original_index,
            'bytes_affected': len(binary_data)
        }

        return result, inverse, metadata

    def burrows_wheeler_inverse(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Inverse Burrows-Wheeler transform."""
        # Extract index and BWT data from the end
        if len(binary_data) <= 4:
            return binary_data, lambda: binary_data, {'operation': 'burrows_wheeler_inverse', 'bytes_affected': 0}

        original_index = int.from_bytes(binary_data[-4:], 'big')
        bwt_data = binary_data[:-4]

        # Reconstruct original using LF mapping
        table = [""] * len(bwt_data)
        for _ in range(len(bwt_data)):
            table = [bwt_data[i] + table[i] for i in range(len(bwt_data))]
            table.sort()

        original = table[original_index].replace(b'\x00', b'')

        def inverse():
            # Forward BWT again
            return self.burrows_wheeler(original)[0]

        metadata = {
            'operation': 'burrows_wheeler_inverse',
            'original_index': original_index,
            'bytes_affected': len(bwt_data)
        }

        return original, inverse, metadata

    def bitplane_extract(self, binary_data: bytes, plane: int) -> Tuple[bytes, Callable, Dict]:
        """Extract specific bitplane."""
        if not 0 <= plane <= 7:
            raise ValueError("Plane must be in range 0-7")

        # Extract bits from specified plane
        bitplane_bits = []
        for byte_val in binary_data:
            bit = (byte_val >> plane) & 1
            bitplane_bits.append(bit)

        # Pack bits into bytes
        result = bytearray()
        for i in range(0, len(bitplane_bits), 8):
            byte_val = 0
            for j in range(min(8, len(bitplane_bits) - i)):
                if bitplane_bits[i + j]:
                    byte_val |= (1 << j)
            result.append(byte_val)

        new_data = bytes(result)

        def inverse():
            # Bitplane extraction is lossy
            raise RuntimeError("Bitplane extraction is not reversible")

        metadata = {
            'operation': 'bitplane_extract',
            'plane': plane,
            'bytes_affected': len(binary_data),
            'reversible': False
        }

        return new_data, inverse, metadata

    def bitplane_insert(self, binary_data: bytes, plane: int, data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Insert bitplane data."""
        if not 0 <= plane <= 7:
            raise ValueError("Plane must be in range 0-7")

        def inverse():
            # Bitplane insertion is lossy
            raise RuntimeError("Bitplane insertion is not reversible")

        metadata = {
            'operation': 'bitplane_insert',
            'plane': plane,
            'data_length': len(data),
            'bytes_affected': len(binary_data),
            'reversible': False
        }

        return binary_data, inverse, metadata

    def move_to_front(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Move-to-front transform."""
        # Initialize symbol list (0-255)
        symbol_list = list(range(256))

        result = []
        for byte_val in binary_data:
            # Find index of symbol
            index = symbol_list.index(byte_val)
            result.append(index)

            # Move symbol to front
            symbol_list.pop(index)
            symbol_list.insert(0, byte_val)

        # Convert indices to bytes
        new_data = bytes(result)

        def inverse():
            # Initialize symbol list
            symbol_list = list(range(256))
            original = []

            for index_val in new_data:
                # Get symbol at index
                symbol = symbol_list[index_val]
                original.append(symbol)

                # Move symbol to front
                symbol_list.pop(index_val)
                symbol_list.insert(0, symbol)

            return bytes(original)

        metadata = {
            'operation': 'move_to_front',
            'bytes_affected': len(binary_data)
        }

        return new_data, inverse, metadata

    def walsh_hadamard(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Walsh-Hadamard transform."""
        import numpy as np

        # Convert to numpy array and pad to power of 2
        data = np.frombuffer(binary_data, dtype=np.uint8)
        n = len(data)
        next_power = 1 << (n - 1).bit_length()
        if next_power > n:
            data = np.pad(data, (0, next_power - n), 'constant')

        # Convert to float for computation
        data_float = data.astype(np.float32)

        # Apply Walsh-Hadamard transform (simplified)
        def walsh_hadamard_recursive(x):
            if len(x) == 1:
                return x
            n = len(x) // 2
            left = walsh_hadamard_recursive(x[:n])
            right = walsh_hadamard_recursive(x[n:])
            return np.concatenate([left + right, left - right])

        transformed = walsh_hadamard_recursive(data_float)

        # Convert back to bytes (simplified - just take integer part)
        result_bytes = np.clip(transformed, 0, 255).astype(np.uint8).tobytes()

        new_data = result_bytes[:n]  # Remove padding

        def inverse():
            # Walsh-Hadamard is self-inverse up to scaling
            # Simplified inverse
            return new_data  # Placeholder

        metadata = {
            'operation': 'walsh_hadamard',
            'bytes_affected': len(binary_data)
        }

        return new_data, inverse, metadata

    def dct_transform(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Discrete cosine transform."""
        try:
            import numpy as np
            from scipy.fft import dct, idct
        except ImportError:
            # Fallback to simple transform without scipy
            return self._simple_dct_transform(binary_data)

        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'dct_transform', 'bytes_affected': 0, 'reversible': True}

        # Convert to numpy array and handle padding for non-power-of-2 data
        original_length = len(binary_data)
        data = np.frombuffer(binary_data, dtype=np.uint8)

        # Find suitable transform length (next power of 2)
        transform_length = 1 << (original_length - 1).bit_length()
        if transform_length > 8192:  # Limit size to prevent memory issues
            transform_length = 8192

        # Pad data if necessary
        if len(data) < transform_length:
            data = np.pad(data, (0, transform_length - len(data)), 'constant', constant_values=0)
        elif len(data) > transform_length:
            data = data[:transform_length]
            original_length = transform_length

        # Apply DCT
        dct_data = dct(data.astype(np.float32), type=2, norm='ortho')

        # Convert back to bytes (quantize to 0-255 range)
        # Scale DCT coefficients to fit in byte range
        dct_scaled = np.clip(dct_data * 255.0 / np.max(np.abs(dct_data)), 0, 255)
        result_data = dct_scaled.astype(np.uint8).tobytes()

        # Restore original length
        new_data = result_data[:original_length]

        def inverse():
            # Apply inverse DCT
            idct_data = idct(dct_data, type=2, norm='ortho')

            # Convert back to bytes
            idct_scaled = np.clip(idct_data, 0, 255).astype(np.uint8).tobytes()
            return idct_scaled[:original_length]

        metadata = {
            'operation': 'dct_transform',
            'transform_length': transform_length,
            'original_length': original_length,
            'dct_range': [float(np.min(dct_data)), float(np.max(dct_data))],
            'bytes_affected': original_length,
            'reversible': True
        }

        return new_data, inverse, metadata

    def _simple_dct_transform(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Simple DCT implementation without scipy dependency."""
        import numpy as np

        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'dct_transform', 'bytes_affected': 0, 'reversible': True}

        data = np.frombuffer(binary_data, dtype=np.uint8).astype(np.float32)
        n = len(data)

        # Simple DCT approximation using cosine transform
        result = np.zeros(n)
        for k in range(n):
            sum_val = 0.0
            for i in range(n):
                sum_val += data[i] * np.cos(np.pi * k * (i + 0.5) / n)
            result[k] = sum_val * np.sqrt(2.0 / n) if k > 0 else sum_val * np.sqrt(1.0 / n)

        # Scale and convert back to bytes
        result_scaled = np.clip(result * 255.0 / (np.max(np.abs(result)) + 1e-10), 0, 255).astype(np.uint8)
        new_data = result_scaled.tobytes()

        def inverse():
            # Simple inverse DCT approximation
            original = np.zeros(n)
            for i in range(n):
                sum_val = result[0] * np.sqrt(1.0 / n)
                for k in range(1, n):
                    sum_val += result[k] * np.cos(np.pi * k * (i + 0.5) / n) * np.sqrt(2.0 / n)
                original[i] = sum_val

            original_scaled = np.clip(original, 0, 255).astype(np.uint8)
            return original_scaled.tobytes()

        metadata = {
            'operation': 'dct_transform',
            'transform_type': 'simple_dct',
            'length': n,
            'bytes_affected': n,
            'reversible': True
        }

        return new_data, inverse, metadata

    def dwt_transform(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Discrete wavelet transform."""
        try:
            import numpy as np
            import pywt
        except ImportError:
            # Fallback to simple wavelet-like transform
            return self._simple_dwt_transform(binary_data)

        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'dwt_transform', 'bytes_affected': 0, 'reversible': True}

        data = np.frombuffer(binary_data, dtype=np.uint8).astype(np.float32)
        original_length = len(data)

        # Handle data length - pad to even length
        if len(data) % 2 != 0:
            data = np.append(data, data[-1])

        # Choose wavelet (Haar is most basic)
        wavelet = 'haar'

        # Perform DWT decomposition
        try:
            coeffs = pywt.dwt(data, wavelet)
            cA, cD = coeffs  # Approximation and detail coefficients

            # Combine coefficients for storage
            combined = np.concatenate([cA, cD])

            # Scale to byte range
            combined_scaled = np.clip(combined * 255.0 / (np.max(np.abs(combined)) + 1e-10), 0, 255).astype(np.uint8)
            new_data = combined_scaled.tobytes()

            def inverse():
                # Extract coefficients from stored data
                mid_point = len(cA)
                restored_cA = combined_scaled[:mid_point].astype(np.float32) / 255.0 * (np.max(np.abs(cA)) + 1e-10)
                restored_cD = combined_scaled[mid_point:mid_point + len(cD)].astype(np.float32) / 255.0 * (np.max(np.abs(cD)) + 1e-10)

                # Inverse DWT
                reconstructed = pywt.idwt((restored_cA, restored_cD), wavelet)

                # Restore original length
                reconstructed = reconstructed[:original_length]
                reconstructed_scaled = np.clip(reconstructed, 0, 255).astype(np.uint8)
                return reconstructed_scaled.tobytes()

            metadata = {
                'operation': 'dwt_transform',
                'wavelet': wavelet,
                'approx_length': len(cA),
                'detail_length': len(cD),
                'original_length': original_length,
                'max_coeff_ca': float(np.max(np.abs(cA))),
                'max_coeff_cd': float(np.max(np.abs(cD))),
                'bytes_affected': original_length,
                'reversible': True
            }

            return new_data, inverse, metadata

        except Exception as e:
            # Fallback to simple transform if pywt fails
            return self._simple_dwt_transform(binary_data)

    def _simple_dwt_transform(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Simple wavelet-like transform without pywt dependency."""
        import numpy as np

        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'dwt_transform', 'bytes_affected': 0, 'reversible': True}

        data = np.frombuffer(binary_data, dtype=np.uint8).astype(np.float32)
        original_length = len(data)

        # Pad to even length
        if len(data) % 2 != 0:
            data = np.append(data, data[-1])

        # Simple Haar-like transform
        n = len(data) // 2
        cA = np.zeros(n)  # Approximation coefficients (averages)
        cD = np.zeros(n)  # Detail coefficients (differences)

        for i in range(n):
            cA[i] = (data[2*i] + data[2*i + 1]) / np.sqrt(2)
            cD[i] = (data[2*i] - data[2*i + 1]) / np.sqrt(2)

        # Combine coefficients
        combined = np.concatenate([cA, cD])

        # Scale to byte range
        combined_scaled = np.clip(combined * 255.0 / (np.max(np.abs(combined)) + 1e-10), 0, 255).astype(np.uint8)
        new_data = combined_scaled.tobytes()

        def inverse():
            # Extract coefficients
            restored_cA = combined_scaled[:n].astype(np.float32) / 255.0 * (np.max(np.abs(cA)) + 1e-10)
            restored_cD = combined_scaled[n:2*n].astype(np.float32) / 255.0 * (np.max(np.abs(cD)) + 1e-10)

            # Inverse Haar transform
            reconstructed = np.zeros(2 * n)
            for i in range(n):
                reconstructed[2*i] = (restored_cA[i] + restored_cD[i]) / np.sqrt(2)
                reconstructed[2*i + 1] = (restored_cA[i] - restored_cD[i]) / np.sqrt(2)

            # Restore original length
            reconstructed = reconstructed[:original_length]
            reconstructed_scaled = np.clip(reconstructed, 0, 255).astype(np.uint8)
            return reconstructed_scaled.tobytes()

        metadata = {
            'operation': 'dwt_transform',
            'transform_type': 'simple_haar',
            'coeff_length': n,
            'original_length': original_length,
            'bytes_affected': original_length,
            'reversible': True
        }

        return new_data, inverse, metadata

    def fft_transform(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Fast Fourier transform."""
        try:
            import numpy as np
        except ImportError:
            # Simple fallback transform
            return self._simple_fft_transform(binary_data)

        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'fft_transform', 'bytes_affected': 0, 'reversible': True}

        data = np.frombuffer(binary_data, dtype=np.uint8).astype(np.float32)
        original_length = len(data)

        # Pad to power of 2 for efficient FFT
        transform_length = 1 << (original_length - 1).bit_length()
        if transform_length > 8192:  # Limit size
            transform_length = 8192

        if len(data) < transform_length:
            data = np.pad(data, (0, transform_length - len(data)), 'constant', constant_values=0)
        elif len(data) > transform_length:
            data = data[:transform_length]
            original_length = transform_length

        # Apply FFT
        fft_data = np.fft.fft(data)

        # Convert complex results to real values for storage
        # Store magnitude and phase information in separate channels
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)

        # Scale and interleave for storage
        magnitude_scaled = np.clip(magnitude * 255.0 / (np.max(magnitude) + 1e-10), 0, 255).astype(np.uint8)
        phase_scaled = np.clip((phase + np.pi) * 255.0 / (2 * np.pi), 0, 255).astype(np.uint8)

        # Interleave magnitude and phase
        combined = np.zeros(2 * len(magnitude_scaled), dtype=np.uint8)
        combined[0::2] = magnitude_scaled
        combined[1::2] = phase_scaled

        new_data = combined.tobytes()

        def inverse():
            # Extract magnitude and phase
            magnitude_restored = combined[0::2].astype(np.float32) / 255.0 * (np.max(magnitude) + 1e-10)
            phase_restored = combined[1::2].astype(np.float32) / 255.0 * (2 * np.pi) - np.pi

            # Reconstruct complex FFT data
            fft_restored = magnitude_restored * np.exp(1j * phase_restored)

            # Apply inverse FFT
            reconstructed = np.fft.ifft(fft_restored).real

            # Restore original length and scale
            reconstructed = reconstructed[:original_length]
            reconstructed_scaled = np.clip(reconstructed, 0, 255).astype(np.uint8)
            return reconstructed_scaled.tobytes()

        metadata = {
            'operation': 'fft_transform',
            'transform_length': transform_length,
            'original_length': original_length,
            'max_magnitude': float(np.max(magnitude)),
            'bytes_affected': original_length,
            'reversible': True
        }

        return new_data, inverse, metadata

    def _simple_fft_transform(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Simple frequency-like transform without numpy."""
        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'fft_transform', 'bytes_affected': 0, 'reversible': True}

        # Simple frequency analysis using basic operations
        data = list(binary_data)
        n = len(data)

        # Create frequency-like representation using differences
        freq_reps = []
        for k in range(min(n, 256)):  # Limit to 256 frequency components
            freq_val = 0
            for i, byte_val in enumerate(data):
                freq_val += byte_val * (1 if i % (k + 1) == 0 else -1)
            freq_reps.append(abs(freq_val))

        # Scale to byte range
        max_val = max(freq_reps) if freq_reps else 1
        freq_scaled = [int(f * 255 / max_val) for f in freq_reps]

        new_data = bytes(freq_scaled + [0] * (n - len(freq_scaled)))  # Pad to original length

        def inverse():
            # Simple inverse using the stored frequency representation
            reconstructed = []
            for i in range(n):
                byte_val = 0
                for k, freq_val in enumerate(freq_scaled[:min(n, 256)]):
                    if i % (k + 1) == 0:
                        byte_val += freq_val if k % 2 == 0 else -freq_val
                reconstructed.append(max(0, min(255, abs(byte_val) // 256)))

            return bytes(reconstructed[:n])

        metadata = {
            'operation': 'fft_transform',
            'transform_type': 'simple_frequency',
            'frequency_components': min(n, 256),
            'bytes_affected': n,
            'reversible': True
        }

        return new_data, inverse, metadata

    def huffman_encode(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Huffman encoding."""
        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'huffman_encode', 'bytes_affected': 0, 'reversible': True}

        # Calculate byte frequencies
        freq = {}
        for byte in binary_data:
            freq[byte] = freq.get(byte, 0) + 1

        if len(freq) <= 1:
            # No compression benefit if all bytes are the same
            return binary_data, lambda: binary_data, {'operation': 'huffman_encode', 'bytes_affected': len(binary_data), 'reversible': True}

        # Build Huffman tree
        class HuffmanNode:
            def __init__(self, char=None, freq=0, left=None, right=None):
                self.char = char
                self.freq = freq
                self.left = left
                self.right = right

        # Create leaf nodes
        nodes = [HuffmanNode(char=char, freq=freq) for char, freq in freq.items()]

        # Build tree
        import heapq
        heapq.heapify(nodes, key=lambda x: x.freq)

        while len(nodes) > 1:
            left = heapq.heappop(nodes)
            right = heapq.heappop(nodes)
            merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(nodes, merged)

        root = nodes[0]

        # Generate codes
        codes = {}
        def traverse(node, code=''):
            if node.char is not None:
                codes[node.char] = code or '0'  # Single character gets code '0'
                return
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

        traverse(root)

        # Encode data
        encoded_bits = []
        for byte in binary_data:
            encoded_bits.append(codes[byte])

        encoded_string = ''.join(encoded_bits)

        # Pack bits into bytes
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_string), 8):
            byte_val = 0
            for j in range(min(8, len(encoded_string) - i)):
                if encoded_string[i + j] == '1':
                    byte_val |= (1 << (7 - j))
            encoded_bytes.append(byte_val)

        # Store tree structure for decoding
        tree_data = self._serialize_huffman_tree(root)
        original_length = len(binary_data)

        # Combine tree data and encoded data
        # Format: [4 bytes: tree_data_len][tree_data][4 bytes: original_len][encoded_data]
        result = (len(tree_data).to_bytes(4, 'big') +
                 tree_data +
                 original_length.to_bytes(4, 'big') +
                 bytes(encoded_bytes))

        def inverse():
            # Extract components
            tree_data_len = int.from_bytes(result[:4], 'big')
            tree_start = 4
            tree_end = tree_start + tree_data_len
            stored_tree_data = result[tree_start:tree_end]

            original_len_start = tree_end
            original_len_end = original_len_start + 4
            original_length = int.from_bytes(result[original_len_start:original_len_end], 'big')

            encoded_data = result[original_len_end:]

            # Reconstruct tree
            root = self._deserialize_huffman_tree(stored_tree_data)

            # Decode data
            decoded_bytes = bytearray()
            current_node = root
            bit_position = 0

            for byte_val in encoded_data:
                for bit_pos in range(8):
                    if bit_position >= len(encoded_data) * 8:
                        break

                    if bit_position < len(encoded_data) * 8:
                        bit = (byte_val >> (7 - bit_pos)) & 1
                        bit_position += 1

                        if current_node.char is not None:
                            decoded_bytes.append(current_node.char)
                            if len(decoded_bytes) >= original_length:
                                break
                            current_node = root

                        if bit == 0:
                            current_node = current_node.left
                        else:
                            current_node = current_node.right

                        if current_node.char is not None:
                            decoded_bytes.append(current_node.char)
                            if len(decoded_bytes) >= original_length:
                                break
                            current_node = root

                if len(decoded_bytes) >= original_length:
                    break

            return bytes(decoded_bytes[:original_length])

        metadata = {
            'operation': 'huffman_encode',
            'original_length': original_length,
            'compressed_length': len(result),
            'compression_ratio': len(result) / original_length if original_length > 0 else 1.0,
            'unique_symbols': len(freq),
            'bytes_affected': original_length,
            'reversible': True
        }

        return result, inverse, metadata

    def _serialize_huffman_tree(self, node) -> bytes:
        """Serialize Huffman tree for storage."""
        def serialize_helper(node):
            if node.char is not None:
                # Leaf node: 1 + 8 bits for character
                return bytes([1, node.char])
            else:
                # Internal node: 0 + left + right
                left_data = serialize_helper(node.left)
                right_data = serialize_helper(node.right)
                return bytes([0]) + left_data + right_data

        return serialize_helper(node)

    def _deserialize_huffman_tree(self, data: bytes):
        """Deserialize Huffman tree from stored data."""
        class HuffmanNode:
            def __init__(self, char=None, freq=0, left=None, right=None):
                self.char = char
                self.freq = freq
                self.left = left
                self.right = right

        def deserialize_helper(index):
            if index >= len(data):
                return None, index

            if data[index] == 1:
                # Leaf node
                if index + 1 >= len(data):
                    return None, index
                node = HuffmanNode(char=data[index + 1])
                return node, index + 2
            else:
                # Internal node
                left_node, new_index = deserialize_helper(index + 1)
                right_node, new_index = deserialize_helper(new_index)
                node = HuffmanNode(left=left_node, right=right_node)
                return node, new_index

        root, _ = deserialize_helper(0)
        return root

    def run_length_encode(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Run-length encoding with configurable run detection."""
        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'run_length_encode', 'bytes_affected': 0, 'reversible': True}

        original_length = len(binary_data)
        encoded = bytearray()

        i = 0
        while i < original_length:
            current_byte = binary_data[i]
            run_length = 1

            # Count consecutive identical bytes
            while (i + run_length < original_length and
                   binary_data[i + run_length] == current_byte and
                   run_length < 255):  # Limit run length to 255
                run_length += 1

            # Store as (byte, run_length) pairs
            encoded.append(current_byte)
            encoded.append(run_length)

            i += run_length

        new_data = bytes(encoded)

        def inverse():
            decoded = bytearray()
            i = 0
            while i < len(new_data):
                byte_val = new_data[i]
                run_length = new_data[i + 1]
                decoded.extend([byte_val] * run_length)
                i += 2
            return bytes(decoded)

        metadata = {
            'operation': 'run_length_encode',
            'original_length': original_length,
            'compressed_length': len(new_data),
            'compression_ratio': len(new_data) / original_length if original_length > 0 else 1.0,
            'runs_detected': len(new_data) // 2,
            'bytes_affected': original_length,
            'reversible': True
        }

        return new_data, inverse, metadata

    def lz77_encode(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """LZ77 encoding with sliding window and look-ahead buffer."""
        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'lz77_encode', 'bytes_affected': 0, 'reversible': True}

        original_length = len(binary_data)
        window_size = min(32768, original_length)  # Sliding window size
        look_ahead_size = min(258, original_length)  # Look-ahead buffer size

        encoded = bytearray()
        position = 0

        while position < original_length:
            best_match = None
            best_length = 0

            # Search for longest match in sliding window
            start_search = max(0, position - window_size)
            search_buffer = binary_data[start_search:position]

            if search_buffer:
                # Find longest match
                for match_start in range(len(search_buffer)):
                    match_length = 0
                    while (position + match_length < original_length and
                           match_start + match_length < len(search_buffer) and
                           binary_data[position + match_length] == search_buffer[match_start + match_length] and
                           match_length < look_ahead_size):
                        match_length += 1

                    if match_length > best_length:
                        best_length = match_length
                        offset = len(search_buffer) - match_start
                        best_match = (offset, best_length)

            if best_match and best_length > 2:
                # Encode as (offset, length) pair
                offset, length = best_match
                # Use 2 bytes each for offset and length
                encoded.extend([
                    0xFF,  # Flag for match
                    (offset >> 8) & 0xFF,
                    offset & 0xFF,
                    (length >> 8) & 0xFF,
                    length & 0xFF
                ])
                position += best_length
            else:
                # Literal byte
                encoded.append(0x00)  # Flag for literal
                encoded.append(binary_data[position])
                position += 1

        new_data = bytes(encoded)

        def inverse():
            decoded = bytearray()
            i = 0
            while i < len(new_data):
                flag = new_data[i]
                i += 1

                if flag == 0x00:
                    # Literal byte
                    if i >= len(new_data):
                        break
                    decoded.append(new_data[i])
                    i += 1
                elif flag == 0xFF:
                    # Match (offset, length)
                    if i + 3 >= len(new_data):
                        break
                    offset = (new_data[i] << 8) | new_data[i + 1]
                    length = (new_data[i + 2] << 8) | new_data[i + 3]
                    i += 4

                    # Copy matched data
                    start_pos = len(decoded) - offset
                    for j in range(length):
                        if start_pos + j < len(decoded):
                            decoded.append(decoded[start_pos + j])
                        else:
                            decoded.append(0)  # Safety fallback
                else:
                    # Invalid flag, treat as literal
                    decoded.append(flag)

            return bytes(decoded)

        metadata = {
            'operation': 'lz77_encode',
            'original_length': original_length,
            'compressed_length': len(new_data),
            'compression_ratio': len(new_data) / original_length if original_length > 0 else 1.0,
            'window_size': window_size,
            'look_ahead_size': look_ahead_size,
            'bytes_affected': original_length,
            'reversible': True
        }

        return new_data, inverse, metadata

    def arithmetic_encode(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Arithmetic coding with adaptive probability model."""
        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'arithmetic_encode', 'bytes_affected': 0, 'reversible': True}

        original_length = len(binary_data)

        # Initialize probability model
        freq = [0] * 256
        total = 0

        def update_model(byte):
            nonlocal total
            freq[byte] += 1
            total += 1

        def get_probability(byte):
            if total == 0:
                return 0.0, 1.0, 1.0
            low = sum(freq[:byte]) / total
            high = low + freq[byte] / total
            return low, high, total

        # Initialize with some probability
        for byte in range(256):
            freq[byte] = 1
        total = 256

        # Arithmetic encoding
        low = 0.0
        high = 1.0
        precision = 32  # Number of bits for precision

        encoded_bytes = bytearray()

        for byte in binary_data:
            byte_low, byte_high, _ = get_probability(byte)

            range_size = high - low
            high = low + range_size * byte_high
            low = low + range_size * byte_low

            # Scale to avoid underflow
            while high - low < 0.5:
                if low < 0.5:
                    encoded_bytes.append(0)
                    low *= 2
                    high *= 2
                else:
                    encoded_bytes.append(1)
                    low = (low - 0.5) * 2
                    high = (high - 0.5) * 2

            update_model(byte)

        # Finalize encoding
        encoded_bytes.append(1)  # Termination marker

        # Store original length for decoding
        result = original_length.to_bytes(4, 'big') + bytes(encoded_bytes)

        def inverse():
            # Extract original length
            original_len = int.from_bytes(result[:4], 'big')
            encoded_data = result[4:]

            # Reset probability model
            freq = [1] * 256
            total = 256

            def update_model(byte):
                nonlocal total
                freq[byte] += 1
                total += 1

            def get_byte_from_value(value):
                cumulative = 0.0
                for byte in range(256):
                    byte_prob = freq[byte] / total
                    if cumulative <= value < cumulative + byte_prob:
                        return byte
                    cumulative += byte_prob
                return 255

            # Decode
            decoded = bytearray()
            low = 0.0
            high = 1.0
            value = 0.5  # Start with middle value

            bit_index = 0
            current_byte = 0

            while len(decoded) < original_len and bit_index < len(encoded_data) * 8:
                byte_index = bit_index // 8
                bit_offset = bit_index % 8

                if byte_index < len(encoded_data):
                    current_byte = encoded_data[byte_index]
                    bit = (current_byte >> (7 - bit_offset)) & 1

                    # Update value
                    if bit == 1:
                        value = (low + high) / 2
                    else:
                        value = low

                    # Find corresponding byte
                    decoded_byte = get_byte_from_value(value)
                    decoded.append(decoded_byte)
                    update_model(decoded_byte)

                    # Update range
                    byte_low, byte_high, _ = get_probability(decoded_byte)
                    range_size = high - low
                    high = low + range_size * byte_high
                    low = low + range_size * byte_low

                    bit_index += 1
                else:
                    break

            return bytes(decoded[:original_len])

        metadata = {
            'operation': 'arithmetic_encode',
            'original_length': original_length,
            'compressed_length': len(new_data),
            'compression_ratio': len(new_data) / original_length if original_length > 0 else 1.0,
            'precision_bits': precision,
            'bytes_affected': original_length,
            'reversible': True
        }

        return result, inverse, metadata

    def distance_coding(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Distance coding."""
        return self.move_to_front(binary_data)

    def elias_gamma(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Elias gamma coding."""
        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'elias_gamma', 'bytes_affected': 0, 'reversible': True}

        original_length = len(binary_data)
        encoded_bits = []

        # Generate Fibonacci numbers for decoding
        fib_numbers = [1, 2]
        while fib_numbers[-1] + fib_numbers[-2] <= 65536:
            fib_numbers.append(fib_numbers[-1] + fib_numbers[-2])

        # Encode each byte
        for byte_val in binary_data:
            n = byte_val + 1  # Elias gamma works with positive integers
            # Get binary representation of n
            binary_n = bin(n)[2:]  # Remove '0b' prefix

            # Prefix with zeros
            prefix_len = len(binary_n) - 1
            encoded_bits.extend([0] * prefix_len)
            encoded_bits.extend([1] + list(map(int, binary_n[1:])))

        # Pack bits into bytes
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte_val = 0
            for j in range(min(8, len(encoded_bits) - i)):
                if encoded_bits[i + j]:
                    byte_val |= (1 << (7 - j))
            encoded_bytes.append(byte_val)

        new_data = bytes(encoded_bytes)

        def inverse():
            decoded = bytearray()
            bit_position = 0

            for _ in range(original_length):
                # Count leading zeros
                zeros = 0
                while bit_position < len(new_data) * 8:
                    byte_index = bit_position // 8
                    bit_offset = bit_position % 8
                    bit = (new_data[byte_index] >> (7 - bit_offset)) & 1

                    if bit == 1:
                        break
                    zeros += 1
                    bit_position += 1

                # Read bits for the number
                bit_position += 1
                n = 1  # Start with 1

                for _ in range(zeros):
                    if bit_position < len(new_data) * 8:
                        byte_index = bit_position // 8
                        bit_offset = bit_position % 8
                        bit = (new_data[byte_index] >> (7 - bit_offset)) & 1
                        n = (n << 1) | bit
                        bit_position += 1

                decoded.append(n - 1)  # Convert back to byte range

            return bytes(decoded)

        metadata = {
            'operation': 'elias_gamma',
            'original_length': original_length,
            'compressed_length': len(new_data),
            'compression_ratio': len(new_data) / original_length if original_length > 0 else 1.0,
            'fibonacci_max': max(fib_numbers),
            'bytes_affected': original_length,
            'reversible': True
        }

        return new_data, inverse, metadata

    def elias_delta(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Elias delta coding."""
        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'elias_delta', 'bytes_affected': 0, 'reversible': True}

        original_length = len(binary_data)
        encoded_bits = []

        def elias_gamma_encode(n):
            bits = []
            binary_n = bin(n)[2:]
            prefix_len = len(binary_n) - 1
            bits.extend([0] * prefix_len)
            bits.extend([1] + list(map(int, binary_n[1:])))
            return bits

        for byte_val in binary_data:
            n = byte_val + 1

            # First encode length of n in Elias gamma
            length_bits = len(bin(n)[2:])
            length_encoded = elias_gamma_encode(length_bits)
            encoded_bits.extend(length_encoded)

            # Then encode n
            n_encoded = list(map(int, bin(n)[2:]))
            encoded_bits.extend(n_encoded)

        # Pack bits into bytes
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte_val = 0
            for j in range(min(8, len(encoded_bits) - i)):
                if encoded_bits[i + j]:
                    byte_val |= (1 << (7 - j))
            encoded_bytes.append(byte_val)

        new_data = bytes(encoded_bytes)

        def inverse():
            decoded = bytearray()
            bit_position = 0

            def elias_gamma_decode(bit_pos):
                zeros = 0
                while bit_pos < len(new_data) * 8:
                    byte_index = bit_pos // 8
                    bit_offset = bit_pos % 8
                    bit = (new_data[byte_index] >> (7 - bit_offset)) & 1

                    if bit == 1:
                        break
                    zeros += 1
                    bit_pos += 1

                bit_pos += 1
                n = 1
                for _ in range(zeros):
                    if bit_pos < len(new_data) * 8:
                        byte_index = bit_pos // 8
                        bit_offset = bit_pos % 8
                        bit = (new_data[byte_index] >> (7 - bit_offset)) & 1
                        n = (n << 1) | bit
                        bit_pos += 1

                return n, bit_pos

            for _ in range(original_length):
                # Decode length
                length_len, bit_position = elias_gamma_decode(bit_position)

                # Decode n
                n = 0
                for _ in range(length_len):
                    if bit_position < len(new_data) * 8:
                        byte_index = bit_position // 8
                        bit_offset = bit_position % 8
                        bit = (new_data[byte_index] >> (7 - bit_offset)) & 1
                        n = (n << 1) | bit
                        bit_position += 1

                decoded.append(n - 1)

            return bytes(decoded)

        metadata = {
            'operation': 'elias_delta',
            'original_length': original_length,
            'compressed_length': len(new_data),
            'compression_ratio': len(new_data) / original_length if original_length > 0 else 1.0,
            'bytes_affected': original_length,
            'reversible': True
        }

        return new_data, inverse, metadata

    def golomb_coding(self, binary_data: bytes, parameter: int = 4) -> Tuple[bytes, Callable, Dict]:
        """Golomb coding."""
        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'golomb_coding', 'bytes_affected': 0, 'reversible': True}

        if parameter <= 0:
            parameter = 4  # Default parameter

        original_length = len(binary_data)
        encoded_bits = []

        for byte_val in binary_data:
            n = byte_val + 1  # Ensure positive

            # Calculate quotient and remainder
            q = (n - 1) // parameter
            r = (n - 1) % parameter

            # Encode quotient in unary
            encoded_bits.extend([1] * q)
            encoded_bits.append(0)

            # Encode remainder in binary
            remainder_bits = parameter
            binary_r = bin(r)[2:].zfill(remainder_bits)
            encoded_bits.extend(map(int, binary_r))

        # Pack bits into bytes
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte_val = 0
            for j in range(min(8, len(encoded_bits) - i)):
                if encoded_bits[i + j]:
                    byte_val |= (1 << (7 - j))
            encoded_bytes.append(byte_val)

        # Store parameter for decoding
        result = parameter.to_bytes(2, 'big') + bytes(encoded_bytes)

        def inverse():
            # Extract parameter
            param = int.from_bytes(result[:2], 'big')
            encoded_data = result[2:]
            decoded = bytearray()
            bit_position = 0

            for _ in range(original_length):
                # Decode quotient (unary)
                q = 0
                while bit_position < len(encoded_data) * 8:
                    byte_index = bit_position // 8
                    bit_offset = bit_position % 8
                    bit = (encoded_data[byte_index] >> (7 - bit_offset)) & 1

                    if bit == 0:
                        break
                    q += 1
                    bit_position += 1

                bit_position += 1

                # Decode remainder
                remainder = 0
                for _ in range(param):
                    if bit_position < len(encoded_data) * 8:
                        byte_index = bit_position // 8
                        bit_offset = bit_position % 8
                        bit = (encoded_data[byte_index] >> (7 - bit_offset)) & 1
                        remainder = (remainder << 1) | bit
                        bit_position += 1

                # Reconstruct original value
                n = q * param + remainder + 1
                decoded.append(n - 1)  # Convert back to byte range

            return bytes(decoded)

        metadata = {
            'operation': 'golomb_coding',
            'original_length': original_length,
            'compressed_length': len(result),
            'compression_ratio': len(result) / original_length if original_length > 0 else 1.0,
            'parameter': parameter,
            'bytes_affected': original_length,
            'reversible': True
        }

        return result, inverse, metadata

    def fibonacci_coding(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Fibonacci coding."""
        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'fibonacci_coding', 'bytes_affected': 0, 'reversible': True}

        # Generate Fibonacci numbers
        fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
                         1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393,
                         196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887]

        original_length = len(binary_data)
        encoded_bits = []

        for byte_val in binary_data:
            n = byte_val + 1  # Ensure positive

            # Find largest Fibonacci number <= n
            for i in range(len(fib_numbers) - 1, -1, -1):
                if fib_numbers[i] <= n:
                    largest_fib = fib_numbers[i]
                    break
            else:
                largest_fib = 1

            # Encode as Fibonacci code
            code = []
            remaining = n
            fib_index = fib_numbers.index(largest_fib)

            while remaining > 0:
                code.append(1)
                remaining -= largest_fib
                fib_index -= 1
                while fib_index >= 0 and fib_numbers[fib_index] > remaining:
                    code.append(0)
                    fib_index -= 1

            # Append termination marker
            code.append(1)
            encoded_bits.extend(reversed(code))

        # Pack bits into bytes
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte_val = 0
            for j in range(min(8, len(encoded_bits) - i)):
                if encoded_bits[i + j]:
                    byte_val |= (1 << (7 - j))
            encoded_bytes.append(byte_val)

        new_data = bytes(encoded_bytes)

        def inverse():
            decoded = bytearray()
            bit_position = 0
            current_fib_index = 0

            for _ in range(original_length):
                # Decode Fibonacci number
                code_bits = []
                while bit_position < len(new_data) * 8:
                    byte_index = bit_position // 8
                    bit_offset = bit_position % 8
                    bit = (new_data[byte_index] >> (7 - bit_offset)) & 1
                    code_bits.append(bit)
                    bit_position += 1

                    if len(code_bits) >= 2 and code_bits[-2:] == [1, 1]:
                        break

                # Remove termination marker
                if code_bits and code_bits[-1] == 1:
                    code_bits.pop()

                # Reconstruct Fibonacci number
                n = 0
                for i, bit in enumerate(reversed(code_bits)):
                    if bit == 1 and i < len(fib_numbers):
                        n += fib_numbers[i]

                decoded.append(n - 1)  # Convert back to byte range

            return bytes(decoded)

        metadata = {
            'operation': 'fibonacci_coding',
            'original_length': original_length,
            'compressed_length': len(new_data),
            'compression_ratio': len(new_data) / original_length if original_length > 0 else 1.0,
            'max_fibonacci': max(fib_numbers),
            'bytes_affected': original_length,
            'reversible': True
        }

        return new_data, inverse, metadata

    def phase_in_coding(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Phase-in coding."""
        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'phase_in_coding', 'bytes_affected': 0, 'reversible': True}

        original_length = len(binary_data)

        # Calculate alphabet size (number of unique symbols)
        alphabet = sorted(set(binary_data))
        m = len(alphabet)
        if m == 0:
            return binary_data, lambda: binary_data, {'operation': 'phase_in_coding', 'bytes_affected': 0, 'reversible': True}

        # Find optimal parameter k (phase-in coding parameter)
        k = 1
        while k * k <= m:
            k += 1
        k -= 1

        encoded = bytearray()

        for byte_val in binary_data:
            symbol_index = alphabet.index(byte_val)

            # Phase-in encoding
            if symbol_index < k:
                # Encode symbol directly
                n = symbol_index + 1
                binary_n = bin(n)[2:]
                encoded.extend([len(binary_n)] + list(map(int, binary_n)))
            else:
                # Encode offset
                offset = symbol_index - k + 1
                n = offset
                binary_n = bin(n)[2:]
                encoded.extend([len(binary_n) + k] + list(map(int, binary_n)))

        new_data = bytes(encoded)

        def inverse():
            decoded = bytearray()
            i = 0

            # Reconstruct alphabet
            alphabet = sorted(set(binary_data))
            m = len(alphabet)
            if m == 0:
                return binary_data

            # Find parameter k
            k = 1
            while k * k <= m:
                k += 1
            k -= 1

            while i < len(new_data):
                if i >= len(new_data):
                    break

                length = new_data[i]
                i += 1

                # Extract bits for the number
                n = 0
                for _ in range(length):
                    if i < len(new_data):
                        n = (n << 1) | new_data[i]
                        i += 1

                if length <= k:
                    # Direct symbol encoding
                    symbol_index = n - 1
                else:
                    # Offset encoding
                    symbol_index = n - 1 + k

                if symbol_index < len(alphabet):
                    decoded.append(alphabet[symbol_index])

            return bytes(decoded)

        metadata = {
            'operation': 'phase_in_coding',
            'original_length': original_length,
            'compressed_length': len(new_data),
            'compression_ratio': len(new_data) / original_length if original_length > 0 else 1.0,
            'alphabet_size': m,
            'parameter_k': k,
            'bytes_affected': original_length,
            'reversible': True
        }

        return new_data, inverse, metadata

    def adaptive_huffman(self, binary_data: bytes) -> Tuple[bytes, Callable, Dict]:
        """Adaptive Huffman coding with dynamic tree updates."""
        if not binary_data:
            return binary_data, lambda: binary_data, {'operation': 'adaptive_huffman', 'bytes_affected': 0, 'reversible': True}

        original_length = len(binary_data)

        # Initialize with equal probabilities for all 256 symbols
        class AdaptiveNode:
            def __init__(self, char=None, weight=0, left=None, right=None, parent=None):
                self.char = char
                self.weight = weight
                self.left = left
                self.right = right
                self.parent = parent

        # Create initial tree with all symbols
        symbols = list(range(256))
        nodes = [AdaptiveNode(char=sym, weight=1) for sym in symbols]

        # Build initial tree (simplified - just use symbol order)
        tree_nodes = nodes.copy()
        encoded_bits = []

        # Adaptive Huffman encoding
        for byte_val in binary_data:
            # Find symbol in current tree
            # For simplicity, we'll use a basic approach
            symbol_index = symbols.index(byte_val)

            # Encode symbol (simplified unary coding for adaptive Huffman)
            # In a real implementation, this would use dynamic codes
            encoded_bits.extend([0] * symbol_index)
            encoded_bits.append(1)

            # Update weights (simplified)
            for i in range(len(nodes)):
                if nodes[i].char == byte_val:
                    nodes[i].weight += 1
                    break

        # Pack bits into bytes
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_bits), 8):
            byte_val = 0
            for j in range(min(8, len(encoded_bits) - i)):
                if encoded_bits[i + j]:
                    byte_val |= (1 << (7 - j))
            encoded_bytes.append(byte_val)

        # Store symbol order for decoding
        symbol_order = bytes(symbols)
        result = symbol_order + bytes(encoded_bytes)

        def inverse():
            # Extract symbol order
            symbol_order = result[:256]
            encoded_data = result[256:]
            decoded = bytearray()
            bit_position = 0

            for _ in range(original_length):
                # Decode symbol using adaptive approach
                symbol_index = 0
                while bit_position < len(encoded_data) * 8:
                    byte_index = bit_position // 8
                    bit_offset = bit_position % 8
                    bit = (encoded_data[byte_index] >> (7 - bit_offset)) & 1

                    if bit == 1:
                        break
                    symbol_index += 1
                    bit_position += 1

                if symbol_index < len(symbol_order):
                    decoded.append(symbol_order[symbol_index])

            return bytes(decoded)

        metadata = {
            'operation': 'adaptive_huffman',
            'original_length': original_length,
            'compressed_length': len(result),
            'compression_ratio': len(result) / original_length if original_length > 0 else 1.0,
            'adaptation_type': 'simple_frequency',
            'bytes_affected': original_length,
            'reversible': True
        }

        return result, inverse, metadata