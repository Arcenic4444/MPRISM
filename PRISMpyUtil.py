from numba import jit, prange, cuda
import math
import numpy as np
import pandas
import itertools
from scipy import stats
from multiprocessing import Pool
from functools import partial
from numba.typed import List

@jit(nopython=True, parallel=True)
def cal_aspect_numba(aspect, Sx, Sy):
    for i in prange(Sx.shape[0]):
        for j in prange(Sy.shape[1]):
            sx = float(Sx[i, j])
            sy = float(Sy[i, j])
            if (sx == 0.0) & (sy == 0.0):
                aspect[i, j] = -1
            elif sx == 0.0:
                if sy > 0.0:
                    aspect[i, j] = 0.0
                else:
                    aspect[i, j] = 180.0
            elif sy == 0.0:
                if sx > 0.0:
                    aspect[i, j] = 90.0
                else:
                    aspect[i, j] = 270.0
            else:
                aspect[i, j] = float(math.atan2(sy, sx) * 57.29578)
                if aspect[i, j] < 0.0:
                    aspect[i, j] = 90.0 - aspect[i, j]
                elif aspect[i, j] > 90.0:
                    aspect[i, j] = 360.0 - aspect[i, j] + 90.0
                else:
                    aspect[i, j] = 90.0 - aspect[i, j]

            if 112.5 > aspect[i, j] >= 67.5:
                aspect[i, j] = 3
            elif 202.5 > aspect[i, j] >= 157.5:
                aspect[i, j] = 5
            elif 292.5 > aspect[i, j] >= 247.5:
                aspect[i, j] = 7
            elif 337.5 > aspect[i, j] >= 292.5:
                aspect[i, j] = 8
            elif 157.5 > aspect[i, j] >= 112.5:
                aspect[i, j] = 4
            elif 67.5 > aspect[i, j] >= 22.5:
                aspect[i, j] = 2
            elif 247.5 > aspect[i, j] >= 202.5:
                aspect[i, j] = 6
            elif aspect[i, j] == -1:
                aspect[i, j] = 0
            else:
                aspect[i, j] = 1

    return aspect


@jit(nopython=True, parallel=True)
def convolution(img_new, padded_array, kernel_size, mode, k):
    n, m = padded_array.shape
    for i in prange(n - kernel_size + 1):
        for j in prange(m - kernel_size + 1):
            a = padded_array[i:i + kernel_size, j:j + kernel_size]
            if mode == 'min':
                img_new[i, j] = (np.min(np.multiply(k, a)))
            if mode == 'max':
                img_new[i, j] = (np.max(np.multiply(k, a)))
            if mode == 'mean':
                img_new[i, j] = (np.mean(np.multiply(k, a)))
    return img_new


@cuda.jit # (nonpython=True)
def convolution_cuda_k(img_new, padded_array, kernel_size, mode, k):
    x, y = cuda.grid(2)
    if x < img_new.shape[0] and y < img_new.shape[1]:
        valmin = 9999
        valmax = 0
        val_mean = 0
        # Calculated in a convolutional kernel
        for i in range(kernel_size):
            for j in range(kernel_size):
                val = padded_array[x + i, y + j] * k[i, j]
                if val < valmin:
                    valmin = val
                if val > valmax:
                    valmax = val
                val_mean += val
        if mode == 0:
            img_new[x, y] = valmin
        if mode == 1:
            img_new[x, y] = valmax
        if mode == 2:
            img_new[x, y] = val_mean / (kernel_size * kernel_size)


def convolution_cuda(img_new, padded_array, kernel_size, mode, k):
    if mode == 'min':
        mode = 0
    if mode == 'max':
        mode = 1
    if mode == 'mean':
        mode = 2
    # ---------------------------------------------allocate streams
    number_of_streams = 1
    segment_size = list()
    for n in img_new.shape:
        segment_size.append(n // number_of_streams)
    # stream_list = list()
    # for i in range(0, number_of_streams * number_of_streams):
    #     stream = cuda.stream()
    #     stream_list.append(stream)
    # ---------------------------------------------allocate the threads
    threads_per_block = (1024, 1024)
    blocks_per_grid = tuple([int(math.ceil(segment_size[m] / threads_per_block[m])) for m in range(len(img_new.shape))])
    streams_out_device = cuda.device_array(segment_size)
    streams_gpu_result = np.empty(img_new.shape)

    # ---------------------------------------------start multi streams, devided to 3*3
    print(segment_size, blocks_per_grid)
    a = 0
    img_new = img_new.astype(np.float64)
    for i in range(0, number_of_streams):
        for j in range(0, number_of_streams):
            a += 1
            img_new_seg = np.ascontiguousarray(img_new[i * segment_size[0]: (i + 1) * segment_size[0],
                                               j * segment_size[1]: (j + 1) * segment_size[1]])
            print(img_new_seg.flags)
            # dimg_new_seg = cuda.to_device(img_new_seg, stream=a)
            dimg_new_seg = cuda.to_device(img_new_seg)
            convolution_cuda_k[threads_per_block, blocks_per_grid](
                dimg_new_seg, padded_array, kernel_size, mode, k)

            streams_gpu_result[i * segment_size[0]: (i + 1) * segment_size[0],
            j * segment_size[1]: (j + 1) * segment_size[1]] = dimg_new_seg.copy_to_host(
            )

    cuda.synchronize()
    return streams_gpu_result


def fft_gages(gage_df):
    gage_id = gage_df.columns
    gage_df = get_datetimecols(gage_df)
    print(gage_df)
    # gage_df = gage_df.mask(gage_df >= 99999, np.nan)
    gage_df.loc[(gage_df[str(gage_id.values[0])] >= 99999), str(gage_id.values[0])] = np.nan

    S1 = gage_df.copy()
    S1 = S1.groupby(['month', 'day'])[str(gage_id.values[0])].apply(np.nanmean)
    S1 = pandas.DataFrame(S1)
    print(S1)
    # daily mean
    S1.index = pandas.date_range('2020-01-01', '2020-12-31')
    S1 = get_datetimecols(S1)

    x = np.array([float(m) for m in S1[gage_id].values])
    # fft daily mean
    S1['data'] = fft_per(x)
    # S1['data'] = x
    # write to new dataframe
    gage_df_smoothed = gage_df.copy()
    for i in S1.iterrows():
        a = gage_df_smoothed.loc[
            (gage_df_smoothed.month == i[1]['month']) & (gage_df_smoothed.day == i[1]['day']), gage_id]
        data = np.ones(len(a)) * i[1]['data']
        data[data == 0] = 1e-13
        gage_df_smoothed.loc[
            (gage_df_smoothed.month == i[1]['month']) & (gage_df_smoothed.day == i[1]['day']), gage_id] = data

    # get the fractions perday
    gage_df_perday_frac = gage_df_smoothed[gage_id].copy()
    gage_df_perday_prop = gage_df_smoothed[gage_id].copy()
    gage_df_perday_frac[gage_id] = gage_df[gage_id].values - gage_df_perday_frac[gage_id].values
    gage_df_perday_prop[gage_id] = gage_df[gage_id].values / gage_df_perday_prop[gage_id].values
    S1 = pandas.DataFrame(S1['data'])
    S1 = S1.rename({'data': str(gage_id.values[0])}, axis=1)

    return S1, gage_df_perday_prop, gage_df_perday_frac


def get_datetimecols(gage_df):
    gage_df['datetime'] = gage_df.index
    gage_df['month'] = pandas.to_datetime(gage_df['datetime']).dt.month
    gage_df['day'] = pandas.to_datetime(gage_df['datetime']).dt.day
    return gage_df


def fft_per(x):
    # Fourier transform
    f = np.fft.fft(x)
    fshift = np.fft.fftshift(f)
    fshift[:179] = 0
    fshift[188:] = 0
    ishift = np.fft.ifftshift(fshift)
    xx = np.abs(np.fft.ifft(ishift))
    return xx


def cooline(x1, x2, y1, y2, data_array):
    dx = math.fabs(x1 - x2)
    dy = math.fabs(y1 - y2)
    if dy == 0:
        return [data_array[y1, x] for x in range(min([x1, x2]), max([x1, x2]) + 1)]
    elif dx == 0:
        return [data_array[y, x1] for y in range(min([y1, y2]), max([y1, y2]) + 1)]
    else:
        k = dy / dx
        xs = np.arange(1, dx)
        ys = (np.round(xs * k) + min([y1, y2])).astype('int')
        result = [data_array[y1, x1], ]
        # append_per_partial = partial(append_per, data_array, ys, xs)
        # with Pool() as P:
        #     result += P.map(append_per_partial, range(len(xs)))
        for m in range(len(xs)):
            result.append(data_array[int(ys[m]), int(xs[m])])
        result.append(data_array[y2, x2])
    return result


def append_per(data_array, ys, xs, m ):
    return data_array[int(ys[m]), int(xs[m])]


def cal_different_facet(x1, x2, y1, y2, data_array):
    cooline_dataset = cooline(x1, x2, y1, y2, data_array)
    upper_bound = cooline_dataset[0] + 1
    if upper_bound > 7:
        upper_bound = 0

    lower_bound = cooline_dataset[0] - 1
    if lower_bound < 0:
        lower_bound = 7

    if lower_bound > upper_bound:
        upper_bound, lower_bound = lower_bound, upper_bound

    diff_count = 0
    diff_count = compare_diff(cooline_dataset, diff_count, lower_bound, upper_bound)
    return diff_count

@jit(nopython=True)
def compare_diff(cooline_dataset, diff_count, lower_bound, upper_bound):
    for cooline_data in cooline_dataset[1:-1]:
        if cooline_data >= upper_bound or cooline_data <= lower_bound:
            diff_count += 1
    return diff_count


if __name__ == '__main__':
    import RasterModule

    # kernel_size = 3 * 2 + 1
    # img_new = np.ones((20000, 30000))
    # padded_array = np.pad(img_new, 3, 'symmetric')
    # print(padded_array)
    # k = np.ones((3 * 2 + 1, 3 * 2 + 1)) * 8
    # convolution_cuda(img_new, padded_array=padded_array, kernel_size=kernel_size, mode=0, k=k)
    dem_fname = f'./DVH_process_dir/dem_64.tif'
    dem = RasterModule.DEM(filename=dem_fname)
    a = dem.aspect
    print(a)
    aa = cooline(1, 666, 2, 4998, a)
    bb = cal_different_facet(1, 666, 2, 4998
                             , a)
    print(bb)
