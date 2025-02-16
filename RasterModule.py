from osgeo import gdal
from VectorModule import *
from scipy.signal import fftconvolve
from multiprocessing import Pool
from PRISMpyUtil import *
from functools import partial


class DEM():
    def __init__(self, dem_array=None, dem_info=None, filename=None):
        if filename != None:
            self.filename = filename
            self.dem_info = self.read_dem()  # 0 dem_array 1 width 2 height 3 bands 4 geotrans 5 proj
            self.dem_array = self.dem_info[0]

        else:
            self.dem_array = dem_array
            self.dem_info = dem_info

        self.resolution = self.dem_info[4][1]
        self.slope, self.Sx, self.Sy = self.cal_slope()
        self.cal_aspect_numba()

    def read_dem(self):
        print('# -------------------------------Importing DEM...-------------------------------#')
        dataset = gdal.Open(self.filename)
        if dataset == None:
            print(self.filename + " file can not be openned!")
            return
        im_width = dataset.RasterXSize
        im_height = dataset.RasterYSize
        im_bands = dataset.RasterCount
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
        im_geotrans = list(dataset.GetGeoTransform())
        im_proj = dataset.GetProjection()
        im_ele = im_data[0:im_height, 0:im_width]
        return im_ele, im_width, im_height, im_bands, im_geotrans, im_proj

    def gaussian_blur(self, size):
        # expand in_array to fit edge of kernel
        padded_array = self.__padarray(size)
        # build kernel
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size)))
        g = (g / g.sum()).astype(self.dem_info[0].dtype)
        # do the Gaussian blur
        return fftconvolve(padded_array, g, mode='valid')

    def cal_slope(self):
        Zbc = self.__padarray(1)
        Sx = (Zbc[1:-1, :-2] - Zbc[1:-1, 2:]) / (2 * self.resolution)  # WE方向
        Sy = (Zbc[2:, 1:-1] - Zbc[:-2, 1:-1]) / (2 * self.resolution)  # NS方向
        slope = np.arctan(np.sqrt(Sx ** 2 + Sy ** 2))
        return slope, Sx, Sy

    def cal_aspect_numba(self):
        aspect = np.ones([self.Sx.shape[0], self.Sx.shape[1]]).astype(np.float32)
        self.aspect = cal_aspect_numba(aspect, self.Sx, self.Sy)

    # after examinition, pool is not quicker than numba
    def cal_aspect_pool(self):
        self.aspect = np.ones([self.Sx.shape[0], self.Sx.shape[1]]).astype(np.float32)
        with Pool() as P:
            P.map(self.aspect_per_pixel, itertools.product(range(self.Sx.shape[0]), range(self.Sy.shape[1])))

    def aspect_per_pixel(self, ij):
        i, j = ij
        sx = float(self.Sx[i, j])
        sy = float(self.Sy[i, j])
        if (sx == 0.0) & (sy == 0.0):
            self.aspect[i, j] = -1
        elif sx == 0.0:
            if sy > 0.0:
                # self.aspect[i, j] = 0.0
                self.aspect[i, j] = 0
            else:
                self.aspect[i, j] = 180.0
        elif sy == 0.0:
            if sx > 0.0:
                self.aspect[i, j] = 90.0
            else:
                self.aspect[i, j] = 270.0
        else:
            self.aspect[i, j] = float(math.atan2(sy, sx) * 57.29578)
            if self.aspect[i, j] < 0.0:
                self.aspect[i, j] = 90.0 - self.aspect[i, j]
            elif self.aspect[i, j] > 90.0:
                self.aspect[i, j] = 360.0 - self.aspect[i, j] + 90.0
            else:
                self.aspect[i, j] = 90.0 - self.aspect[i, j]

        if 112.5 > self.aspect[i, j] >= 67.5:
            self.aspect[i, j] = 3
        elif 202.5 > self.aspect[i, j] >= 157.5:
            self.aspect[i, j] = 5
        elif 292.5 > self.aspect[i, j] >= 247.5:
            self.aspect[i, j] = 7
        elif 337.5 > self.aspect[i, j] >= 292.5:
            self.aspect[i, j] = 8
        elif 157.5 > self.aspect[i, j] >= 112.5:
            self.aspect[i, j] = 4
        elif 67.5 > self.aspect[i, j] >= 22.5:
            self.aspect[i, j] = 2
        elif 247.5 > self.aspect[i, j] >= 202.5:
            self.aspect[i, j] = 6
        elif self.aspect[i, j] == -1:
            self.aspect[i, j] = 0
        else:
            self.aspect[i, j] = 1


    def __padarray(self, size):
        padded_array = np.pad(self.dem_array, size, 'symmetric')
        return padded_array

    @staticmethod
    def write_raster_file(im_geotrans, im_proj, path, input_array):
        if 'int8' in input_array.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in input_array.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(input_array.shape) == 3:
            im_bands, im_height, im_width = input_array.shape
        elif len(input_array.shape) == 2:
            im_bands, (im_height, im_width) = 1, input_array.shape
            input_array = np.array([input_array])
        else:
            im_bands, (im_height, im_width) = 1, input_array.shape


        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
        if (dataset != None):
            dataset.SetGeoTransform(im_geotrans)  
            dataset.SetProjection(im_proj)  
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(input_array[i])
        del dataset
