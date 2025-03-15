import numpy as np
import torch
from tonic.slicers import (
    slice_events_by_time,
)
from typing import Any, List, Tuple


def custom_to_voxel_grid_numpy(events, sensor_size, n_time_bins=10):
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    Implements the event volume from Zhu et al. 2019, Unsupervised event-based learning of optical
    flow, depth, and egomotion.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H].
        n_time_bins: number of bins in the temporal axis of the voxel grid.

    Returns:
        numpy array of n event volumes (n,w,h,t)
    """
    assert "x" and "y" and "t" and "p" in events.dtype.names
    assert sensor_size[2] == 2

    voxel_grid = np.zeros((n_time_bins, sensor_size[1], sensor_size[0]), float).ravel()
    
    # normalize the event timestamps so that they lie between 0 and n_time_bins
    time_diff = events["t"][-1] - events["t"][0]
    if time_diff < 1e-6:  
        ts = np.zeros_like(events["t"], dtype=float)
    else:
        ts = n_time_bins * (events["t"].astype(float) - events["t"][0]) / time_diff

    xs = events["x"].astype(int)
    ys = events["y"].astype(int)
    pols = events["p"]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + tis[valid_indices] * sensor_size[0] * sensor_size[1],
        vals_left[valid_indices],
    )

    valid_indices = (tis + 1) < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + (tis[valid_indices] + 1) * sensor_size[0] * sensor_size[1],
        vals_right[valid_indices],
    )

    voxel_grid = np.reshape(
        voxel_grid, (n_time_bins, 1, sensor_size[1], sensor_size[0])
    )

    return voxel_grid

class SliceByTimeEventsTargets:
    """
    Modified from tonic.slicers.SliceByTimeEventsTargets in the Tonic Library

    Slices an event array along fixed time window and overlap size. The number of bins depends
    on the length of the recording. Targets are copied.

    >        <overlap>
    >|    window1     |
    >        |   window2     |

    Parameters:
        time_window (int): time for window length (same unit as event timestamps)
        overlap (int): overlap (same unit as event timestamps)
        include_incomplete (bool): include the last incomplete slice that has shorter time
    """

    def __init__(self,time_window, overlap=0.0, seq_length=30, seq_stride=15, include_incomplete=False) -> None:
        self.time_window= time_window
        self.overlap= overlap
        self.seq_length=seq_length
        self.seq_stride=seq_stride
        self.include_incomplete=include_incomplete

    def slice(self, data: np.ndarray, targets: int) -> List[np.ndarray]:
        metadata = self.get_slice_metadata(data, targets)
        return self.slice_with_metadata(data, targets, metadata)

    def get_slice_metadata(
        self, data: np.ndarray, targets: int
    ) -> List[Tuple[int, int]]:
        t = data["t"]
        stride = self.time_window - self.overlap
        assert stride > 0

        if self.include_incomplete:
            n_slices = int(np.ceil(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        else:
            n_slices = int(np.floor(((t[-1] - t[0]) - self.time_window) / stride) + 1)
        n_slices = max(n_slices, 1)  # for strides larger than recording time

        window_start_times = np.arange(n_slices) * stride + t[0]
        window_end_times = window_start_times + self.time_window
        indices_start = np.searchsorted(t, window_start_times)[:n_slices]
        indices_end = np.searchsorted(t, window_end_times)[:n_slices]

        if not self.include_incomplete:
            # get the strided indices for loading labels
            label_indices_start = np.arange(0, targets.shape[0]-self.seq_length, self.seq_stride)
            label_indices_end = label_indices_start + self.seq_length
        else:
            label_indices_start = np.arange(0, targets.shape[0], self.seq_stride)
            label_indices_end = label_indices_start + self.seq_length
            # the last label indices end should be the last label
            label_indices_end[-1] = targets.shape[0]

        assert targets.shape[0] >= label_indices_end[-1]

        return list(zip(zip(indices_start, indices_end), zip(label_indices_start, label_indices_end)))

    @staticmethod
    def slice_with_metadata(
        data: np.ndarray, targets: int, metadata: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ):
        return_data = []
        return_target = []
        for tuple1, tuple2 in metadata:
            return_data.append(data[tuple1[0]:tuple1[1]])
            return_target.append(targets[tuple2[0]:tuple2[1]])

        return return_data, return_target


class SliceLongEventsToShort:
    def __init__(self, time_window, overlap, include_incomplete):
        """
        Initialize the transformation.

        Args:
        - time_window (int): The length of each sub-sequence.
        """
        self.time_window = time_window
        self.overlap = overlap
        self.include_incomplete = include_incomplete

    def __call__(self, events):
        return slice_events_by_time(events, self.time_window, self.overlap, self.include_incomplete)


class EventSlicesToVoxelGrid:
    def __init__(self, sensor_size, n_time_bins, per_channel_normalize):
        """
        Initialize the transformation.

        Args:
        - sensor_size (tuple): The size of the sensor.
        - n_time_bins (int): The number of time bins.
        """
        self.sensor_size = sensor_size
        self.n_time_bins = n_time_bins
        self.per_channel_normalize = per_channel_normalize

    def __call__(self, event_slices):
        """
        Apply the transformation to the given event slices.

        Args:
        - event_slices (Tensor): The input event slices.

        Returns:
        - Tensor: A batched tensor of voxel grids.
        """
        voxel_grids = []
        for event_slice in event_slices:
            voxel_grid = custom_to_voxel_grid_numpy(event_slice, self.sensor_size, self.n_time_bins)
            voxel_grid = voxel_grid.squeeze(-3)
            if self.per_channel_normalize:
                # Calculate mean and standard deviation only at non-zero values
                non_zero_entries = (voxel_grid != 0)
                for c in range(voxel_grid.shape[0]):
                    mean_c = voxel_grid[c][non_zero_entries[c]].mean()
                    std_c = voxel_grid[c][non_zero_entries[c]].std()

                    voxel_grid[c][non_zero_entries[c]] = (voxel_grid[c][non_zero_entries[c]] - mean_c) / (std_c + 1e-10)
            voxel_grids.append(voxel_grid)
        return np.array(voxel_grids).astype(np.float32)


class SplitSequence:
    def __init__(self, sub_seq_length, stride):
        """
        Initialize the transformation.

        Args:
        - sub_seq_length (int): The length of each sub-sequence.
        - stride (int): The stride between sub-sequences.
        """
        self.sub_seq_length = sub_seq_length
        self.stride = stride

    def __call__(self, sequence, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - sequence (Tensor): The input sequence of frames.
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of sub-sequences.
        - Tensor: A batched tensor of corresponding labels.
        """

        sub_sequences = []
        sub_labels = []

        for i in range(0, len(sequence) - self.sub_seq_length + 1, self.stride):
            sub_seq = sequence[i:i + self.sub_seq_length]
            sub_seq_labels = labels[i:i + self.sub_seq_length]
            sub_sequences.append(sub_seq)
            sub_labels.append(sub_seq_labels)

        return np.stack(sub_sequences), np.stack(sub_labels)
    

class SplitLabels:
    def __init__(self, sub_seq_length, stride):
        """
        Initialize the transformation.

        Args:
        - sub_seq_length (int): The length of each sub-sequence.
        - stride (int): The stride between sub-sequences.
        """
        self.sub_seq_length = sub_seq_length
        self.stride = stride
        # print(f"stride is {self.stride}")

    def __call__(self, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        sub_labels = []
        
        for i in range(0, len(labels) - self.sub_seq_length + 1, self.stride):
            sub_seq_labels = labels[i:i + self.sub_seq_length]
            sub_labels.append(sub_seq_labels)

        return np.stack(sub_labels)

class ScaleLabel:
    def __init__(self, scaling_factor):
        """
        Initialize the transformation.

        Args:
        - scaling_factor (float): How much the spatial scaling was done on input
        """
        self.scaling_factor = scaling_factor


    def __call__(self, labels):
        """
        Apply the transformation to the given sequence and labels.

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        labels[:,:2] =  labels[:,:2] * self.scaling_factor
        return labels
    
class LabelTemporalSubsample:
    def __init__(self, temporal_subsample_factor):
        self.temp_subsample_factor = temporal_subsample_factor

    def __call__(self, labels):
        """
        temorally subsample the labels
        """
        interval = int(1/self.temp_subsample_factor)
        return labels[::interval]
    

class NormalizeLabel:
    def __init__(self, pseudo_width, pseudo_height):
        """
        Initialize the transformation.

        Args:
        - scaling_factor (float): How much the spatial scaling was done on input
        """
        self.pseudo_width = pseudo_width
        self.pseudo_height = pseudo_height
    
    def __call__(self, labels):
        """
        Apply normalization on label, with pseudo width and height

        Args:
        - labels (Tensor): The corresponding labels.

        Returns:
        - Tensor: A batched tensor of corresponding labels.
        """
        labels[:, 0] = labels[:, 0] / self.pseudo_width
        labels[:, 1] = labels[:, 1] / self.pseudo_height
        return labels


class SpatialShift:
    def __init__(self, max_shift_x, max_shift_y, sensor_size):
        """
        Initialize the transformation on voxel grid.

        Args:
            max_shift_x (int): Maximum shift along x-axis.
            max_shift_y (int): Maximum shift along y-axis.
            sensor_size (tuple): (width, height) tham khảo (không nhất thiết phải khớp với kích thước thực tế của voxel grid).
        """
        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y
        # sensor_size tham khảo, nhưng trong __call__ ta sẽ lấy kích thước thực tế của voxel grid.
        self.ref_sensor_width, self.ref_sensor_height = int(sensor_size[0]), int(sensor_size[1])

    def __call__(self, voxel_grid):
        """
        Apply a random spatial shift on voxel grid data.
        Hỗ trợ voxel grid dạng 3D (n_time_bins, height, width) hoặc 4D (n_time_bins, channels, height, width).

        Args:
            voxel_grid (np.ndarray): Voxel grid data.

        Returns:
            np.ndarray: Shifted voxel grid.
        """
        shift_x = np.random.randint(-self.max_shift_x, self.max_shift_x + 1)
        shift_y = np.random.randint(-self.max_shift_y, self.max_shift_y + 1)

        # Xác định kích thước thực tế của voxel grid dựa trên số chiều
        if voxel_grid.ndim == 3:
            # voxel_grid shape: (n_time_bins, height, width)
            height = voxel_grid.shape[1]
            width = voxel_grid.shape[2]
            shifted = np.zeros_like(voxel_grid)
            
            if shift_y >= 0:
                src_y = slice(0, height - shift_y)
                dst_y = slice(shift_y, height)
            else:
                src_y = slice(-shift_y, height)
                dst_y = slice(0, height + shift_y)
                
            if shift_x >= 0:
                src_x = slice(0, width - shift_x)
                dst_x = slice(shift_x, width)
            else:
                src_x = slice(-shift_x, width)
                dst_x = slice(0, width + shift_x)
            
            shifted[:, dst_y, dst_x] = voxel_grid[:, src_y, src_x]
            
        elif voxel_grid.ndim == 4:
            # voxel_grid shape: (n_time_bins, channels, height, width)
            height = voxel_grid.shape[2]
            width = voxel_grid.shape[3]
            shifted = np.zeros_like(voxel_grid)
            
            if shift_y >= 0:
                src_y = slice(0, height - shift_y)
                dst_y = slice(shift_y, height)
            else:
                src_y = slice(-shift_y, height)
                dst_y = slice(0, height + shift_y)
                
            if shift_x >= 0:
                src_x = slice(0, width - shift_x)
                dst_x = slice(shift_x, width)
            else:
                src_x = slice(-shift_x, width)
                dst_x = slice(0, width + shift_x)
            
            shifted[:, :, dst_y, dst_x] = voxel_grid[:, :, src_y, src_x]
        else:
            raise ValueError("Voxel grid must be a 3D or 4D numpy array.")
        
        return shifted
    
class EventCutout:
    def __init__(self, cutout_width, cutout_height, sensor_size):
        """
        Khởi tạo phép biến đổi EventCutout.

        Args:
            cutout_width (int): Chiều rộng của vùng cutout.
            cutout_height (int): Chiều cao của vùng cutout.
            sensor_size (tuple): (width, height) của cảm biến. (Được dùng cho raw events)
        """
        self.cutout_width = cutout_width
        self.cutout_height = cutout_height
        self.sensor_width, self.sensor_height = int(sensor_size[0]), int(sensor_size[1])

    def __call__(self, events):
        """
        Áp dụng phép cutout cho dữ liệu sự kiện.
        
        - Nếu events là raw events (structured array), thực hiện cắt dựa trên các trường "x", "y".
        - Nếu events là voxel grid (plain numpy array 3D hoặc 4D), sẽ zero out một vùng chữ nhật ngẫu nhiên.

        Args:
            events (np.ndarray): Mảng sự kiện, có thể là structured array hoặc voxel grid.

        Returns:
            np.ndarray: Dữ liệu sau khi cắt (loại bỏ hoặc zero-out vùng).
        """
        # Kiểm tra xem events có phải là structured array hay không
        if events.dtype.names is not None:
            # Xử lý raw events
            x_min = np.random.randint(0, self.sensor_width - self.cutout_width + 1)
            y_min = np.random.randint(0, self.sensor_height - self.cutout_height + 1)
            x_max = x_min + self.cutout_width
            y_max = y_min + self.cutout_height

            mask = (events["x"] < x_min) | (events["x"] >= x_max) | (events["y"] < y_min) | (events["y"] >= y_max)
            return events[mask]
        else:
            # Giả sử đây là voxel grid. Hỗ trợ cả mảng 3D và 4D.
            if events.ndim == 3:
                # Voxel grid có shape: (n_time_bins, height, width)
                H = events.shape[1]
                W = events.shape[2]
                x_min = np.random.randint(0, W - self.cutout_width + 1)
                y_min = np.random.randint(0, H - self.cutout_height + 1)
                x_max = x_min + self.cutout_width
                y_max = y_min + self.cutout_height
                events[:, y_min:y_max, x_min:x_max] = 0
                return events
            elif events.ndim == 4:
                # Voxel grid có shape: (n_time_bins, channels, height, width)
                H = events.shape[2]
                W = events.shape[3]
                x_min = np.random.randint(0, W - self.cutout_width + 1)
                y_min = np.random.randint(0, H - self.cutout_height + 1)
                x_max = x_min + self.cutout_width
                y_max = y_min + self.cutout_height
                events[:, :, y_min:y_max, x_min:x_max] = 0
                return events
            else:
                raise ValueError("Unexpected input dimension for EventCutout.")


class TemporalShift:
    def __init__(self, max_shift=2, mode="wrap"):
        """
        Khởi tạo phép biến đổi Temporal Shift.

        Args:
            max_shift (int): Độ lệch tối đa theo trục thời gian.
            mode (str): Cách xử lý phần dư sau khi dịch chuyển, có thể là:
                        - "wrap": Xoay vòng dữ liệu (các sự kiện bị đẩy ra ngoài quay lại đầu)
                        - "zero": Điền phần bị đẩy ra ngoài bằng 0
        """
        self.max_shift = max_shift
        self.mode = mode

    def __call__(self, events):
        """
        Áp dụng Temporal Shift lên dữ liệu.

        Args:
            events (np.ndarray): Dữ liệu có thể là structured array (raw events) hoặc voxel grid.

        Returns:
            np.ndarray: Dữ liệu sau khi dịch chuyển.
        """
        if events.dtype.names is not None:
            # Xử lý raw events
            shift_t = np.random.randint(-self.max_shift, self.max_shift + 1)
            events["t"] += shift_t
            events["t"] = np.clip(events["t"], 0, None)  # Đảm bảo t >= 0
            return events
        else:
            # Xử lý voxel grid (T, H, W) hoặc (T, C, H, W)
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            if shift == 0:
                return events
            
            if events.ndim == 3:
                # Voxel grid (T, H, W)
                if self.mode == "wrap":
                    return np.roll(events, shift, axis=0)
                else:  # mode == "zero"
                    if shift > 0:
                        events[shift:] = events[:-shift]
                        events[:shift] = 0
                    else:
                        events[:shift] = events[-shift:]
                        events[shift:] = 0
            elif events.ndim == 4:
                # Voxel grid (T, C, H, W)
                if self.mode == "wrap":
                    return np.roll(events, shift, axis=0)
                else:
                    if shift > 0:
                        events[shift:] = events[:-shift]
                        events[:shift] = 0
                    else:
                        events[:shift] = events[-shift:]
                        events[shift:] = 0
            return events

