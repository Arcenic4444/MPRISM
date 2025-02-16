from RasterModule import *
from VectorModule import *


class FactorsCalculator():
    """
    this class is for calculate the factors used in PRISM and connect it with in-situ points
    dem: the dimension and resolution must be same in x and y directions
        gaussian filter is only avalible for pixels currently
    """

    def __init__(self, dem_filename, blur_steps, fork_file, gage_dir, gene_fork_file, gene_gage_dir,
                 influence_radius=50000, minimum_influence_radius=7000, elevation_weighting_exponent=1.0,
                 minimum_elevation_difference=200.0, maximum_elevation_difference=1500, facet_weighting_exponent=1.0,
                 minimum_facet_weighting=0.000000, minimum_elevation_weighting=0.000000,
                 distance_weighting_exponent=0.5, minimum_cluster_weighting=0.000000,
                 terrain_threshold_2d=50, terrain_threshold_3d=500, effective_index_radius_small=20000,
                 effective_index_radius_large=50000, posi_nega_terrain_radius=30000, maximum_influence_radius=14000,
                 posi_nega_terrain_weighting_exponent=0.125):

        self.DEM_PRECISION = 50  # assume the precision of dem is 50m
        self.INFLUENCE_RADIUS = influence_radius  # 30-50 km.
        self.MINIMUM_INFLUENCE_RADIUS = minimum_influence_radius
        self.ELEVATION_WEIGHTING_EXPONENT = elevation_weighting_exponent
        self.MINIMUM_ELEVATION_DIFFERENCE = minimum_elevation_difference
        self.MAXIMUM_ELEVATION_DIFFERENCE = maximum_elevation_difference
        self.FACET_WEIGHTING_EXPONENT = facet_weighting_exponent
        self.MINIMUM_FACET_WEIGHTING = minimum_facet_weighting
        self.MINIMUM_ELEVATION_WEIGHTING = minimum_elevation_weighting
        self.DISTANCE_WEIGHTING_EXPONENT = distance_weighting_exponent
        self.POSI_NAGE_TERRAIN_WEIGHTING_EXPONENT = posi_nega_terrain_weighting_exponent
        self.MINIMUM_CLUSTER_WEIGHTING = minimum_cluster_weighting
        self.TERRAIN_THRESHOLD_2D = terrain_threshold_2d
        self.TERRAIN_THRESHOLD_3D = terrain_threshold_3d
        # self.MINIMUM_VALID_REGRESSION = minimum_valid_regression_slope
        # self.MAXIMUM_VALID_REGRESSION = maximum_valid_regression_slope
        # self.DEFAULT_VALID_REGRESSION = default_valid_regression_slope
        self.EFFECTIVE_INDEX_RADIUS_SMALL = effective_index_radius_small
        self.EFFECTIVE_INDEX_RADIUS_LARGE = effective_index_radius_large
        self.POSI_NAGE_TERRAIN_RADIUS = posi_nega_terrain_radius
        self.MAXIMUM_INFLUENCE_RADIUS = maximum_influence_radius

        # --------------------------------------------1. topographic-------------------------
        self.dem0 = DEM(filename=dem_filename)
        self.dems = [self.dem0, ]
        self.base_resolution = self.dem0.resolution
        self.blur_steps = blur_steps
        self.blur_topography()
        # --------------------------------------------2. cal_effective_terrain_index---------
        self.effective_terrain_index = self.cal_effective_terrain_index()
        # --------------------------------------------2. cal_posi_nega_terrain_index---------
        self.posi_nega_terrain_discrete = self.cal_posi_nega_terrain()
        # --------------------------------------------3. vectors----in-situ and generated pnts---------------------
        self.insitu_pcps = PcpGagePnts(fork_file, gage_dir)
        self.generated_pcps = PcpGenePnts(gene_fork_file, gene_gage_dir)
        self.all_forks = pandas.concat([self.insitu_pcps.fork_df, self.generated_pcps.fork_df],
                                       ignore_index=True)

        # --------------1.1 bridge-------------------------
        self.insitu_pcps.fork_df = self.bridge_vector_raster(self.insitu_pcps.fork_df)

    def blur_topography(self):
        """
        define blur steps, then you will get filtered DEM and the corresponding aspect and slope informations.
        :return:
        """
        # try:
        # with Pool() as P:
        #     P.map(self.per_blur_topography, self.blur_steps)
        # except AssertionError:
        #     print('called multiprocessing failed, calling threads')
        #     from concurrent.futures import ThreadPoolExecutor
        #     P = ThreadPoolExecutor(len(self.blur_steps))
        #     for blur in self.blur_steps:
        #         P.submit(self.per_blur_topography,blur)

        # map(self.per_blur_topography, self.blur_steps)
        for blur in self.blur_steps:
            self.per_blur_topography(blur)

    def per_blur_topography(self, blur):
        print(
            f'# -------------------------------Performing Gaussian filter: step-{str(blur)}-------------------------------#')
        print(f'# --------------Checking for weather there already exists DEM and aspect files----------------#')

        dem_fname = f'./01-DEM-Aspect/dem_{str(blur)}.tif'
        aspect_fname = f'./01-DEM-Aspect/aspect_{str(blur)}.tif'
        if os.path.isfile(dem_fname):
            print(f'# --------------DEM and aspect files already exist, importing...----------------#')
            exec(f'self.dem{str(blur)} = DEM(filename=dem_fname)')
        else:
            if not os.path.exists(r'./01-DEM-Aspect'):
                os.makedirs(r'./01-DEM-Aspect')
            print(f'# --------------DEM and aspect files do not exist, calculating...----------------#')
            exec(f'self.dem{str(blur)} = DEM(dem_array=self.dem0.gaussian_blur(blur),dem_info=self.dem0.dem_info)')
            exec(
                f"self.dem{str(blur)}.write_raster_file(input_array = self.dem{str(blur)}.dem_array,"
                f"im_geotrans = self.dem0.dem_info[4],im_proj = self.dem0.dem_info[5],"
                f"path = dem_fname)")
            exec(
                f"self.dem{str(blur)}.write_raster_file(input_array = self.dem{str(blur)}.aspect,"
                f"im_geotrans = self.dem0.dem_info[4],im_proj = self.dem0.dem_info[5],"
                f"path = aspect_fname)")
        exec(f'self.dems.append(self.dem{str(blur)})')

    def cluster_weighting(self, fork_df):
        print(
            f'# -------------------------------1.1 Calculating cluster weights...-------------------------------#')
        fork_df['Wc'] = [0] * len(fork_df)
        fork_df['Wc'] = fork_df['Wc'].astype('object')

        wc = []
        for i in self.insitu_pcps.fork_df.iterrows():
            vi = 0
            hi = 0
            for j in self.insitu_pcps.fork_df.iterrows():
                if j[0] != i[0]:
                    # ---------------------------------------------------vi
                    sij = self.__cluster_cal_s(j[1]['dem0'], i[1]['dem0'])
                    if sij >= 0 and sij <= self.DEM_PRECISION:
                        vi += (self.DEM_PRECISION - sij) / self.DEM_PRECISION
                    else:
                        vi += 0
                    # ---------------------------------------------------hi
                    dij = self.__cluster_cal_distance(j[1]['LONG'], j[1]['LAT'], i[1]['LONG'], i[1]['LAT'])
                    if dij >= 0 and dij <= 0.2 * self.INFLUENCE_RADIUS:
                        hi += (0.2 * self.INFLUENCE_RADIUS - dij) / (0.2 * self.INFLUENCE_RADIUS)
                    else:
                        hi += 0
                else:
                    continue
            Sc = vi * hi
            if Sc > 1:
                wc.append(1 / Sc)
            else:
                wc.append(1)
        aa = 0
        for i in fork_df.iterrows():
            Wc = []
            cc = 0
            for j in self.insitu_pcps.fork_df.iterrows():
                if i[1]['id'] != j[1]['id']:
                    Wc.append(wc[cc])
                    cc += 1

            fork_df.at[aa, 'Wc'] = self.weight_normalization(Wc)
            aa += 1
        return fork_df

    def __cluster_cal_distance(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])  # Latitude and longitude converted to radians
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        distance = 2 * math.asin(math.sqrt(a)) * 6371 * 1000  # Mean radius of the Earth, 6371km
        distance = round(distance, 3)
        return distance

    def __cluster_cal_s(self, ele1, ele2):
        if math.fabs(ele1 - ele2) > self.DEM_PRECISION:
            return math.fabs(ele1 - ele2) - self.DEM_PRECISION
        else:
            return 0

    def bridge_vector_raster(self, input_forkdf):
        print('----------------------bridging vectors and rasters---------------------')
        # -------------------bridge elevation and aspect-----------------
        input_forkdf = self.__findraster_coor(input_forkdf)
        for dem in enumerate(self.dems):
            input_forkdf_dem_data = self.__bridge_vector_raster_findraster(input_forkdf, dem[1].dem_array)
            input_forkdf['dem' + str(dem[0])] = input_forkdf_dem_data
            input_forkdf_aspect_data = self.__bridge_vector_raster_findraster(input_forkdf, dem[1].aspect)
            input_forkdf['aspect' + str(dem[0])] = input_forkdf_aspect_data
        # -------------------bridge effective terrain index -----------------
        input_forkdf_i3d_data = self.__bridge_vector_raster_findraster(input_forkdf, self.effective_terrain_index)
        input_forkdf_PNTd_data = self.__bridge_vector_raster_findraster(input_forkdf, self.posi_nega_terrain_discrete)
        input_forkdf['effective_terrain_index'] = input_forkdf_i3d_data
        input_forkdf['posi_nega_terrain_discrete'] = input_forkdf_PNTd_data

        return input_forkdf

    def __bridge_vector_raster_findraster(self, pntdf, ds_array):
        """
        this function is used for extracting raster data to points
        :return:
        """
        return [ds_array[m] for m in zip(pntdf['raster_y'], pntdf['raster_x'])]

    def __findraster_coor(self, pntdf):
        """
        this function is used for extracting raster data to points
        :return:
        """
        xOrigin = self.dem0.dem_info[4][0]
        yOrigin = self.dem0.dem_info[4][3]
        pixelWidth = self.dem0.dem_info[4][1]
        pixelHeight = self.dem0.dem_info[4][5]

        pntdf['raster_x'] = [0] * len(pntdf)
        pntdf['raster_y'] = [0] * len(pntdf)

        for i in pntdf.iterrows():
            x = pntdf.loc[i[0], 'LONG']
            y = pntdf.loc[i[0], 'LAT']
            xOffset = int((x - xOrigin) / pixelWidth)
            yOffset = int((y - yOrigin) / pixelHeight)
            try:
                pntdf.loc[i[0], 'raster_x'] = xOffset
                pntdf.loc[i[0], 'raster_y'] = yOffset
            except IndexError:
                pntdf.loc[i[0], 'raster_x'] = np.nan
                pntdf.loc[i[0], 'raster_y'] = np.nan
        return pntdf

    def distance_weighting(self, fork_df):
        """
        we do not confine it to in-situ stations or generated cells so that we can change the grids we want to apply the
         method(all_df) or just for evaluating(in-situ only), but one should notice that no matter what grids we want to
         perform the interpolation, the corresponding regression is based on in-situ gages, the weights will change because
         the related distance change with the positions of cells which are waiting for being interpolated.
        :param fork_df:
        :return:
        """
        print(
            f'# -------------------------------1.2 Calculating distance weights...-------------------------------#')
        fork_df['Wd'] = [0] * len(fork_df)
        fork_df['Wd'] = fork_df['Wd'].astype('object')
        for i in fork_df.iterrows():
            Wd = []
            for j in self.insitu_pcps.fork_df.iterrows():
                if j[1]['id'] != i[1]['id']:
                    d = self.__cluster_cal_distance(j[1]['LONG'], j[1]['LAT'], i[1]['LONG'], i[1]['LAT'])
                    if d <= self.MINIMUM_INFLUENCE_RADIUS:
                        Wd.append((1 / self.MINIMUM_INFLUENCE_RADIUS) ** self.DISTANCE_WEIGHTING_EXPONENT)
                    else:
                        # self.MAXIMUM_INFLUENCE_RADIUS >= d > self.MINIMUM_INFLUENCE_RADIUS
                        Wd.append((1 / d) ** self.DISTANCE_WEIGHTING_EXPONENT)
                    # else:
                    #     Wd.append((1 / self.MAXIMUM_INFLUENCE_RADIUS) ** 2)
                else:
                    continue
            fork_df.at[i[0], 'Wd'] = self.weight_normalization(
                Wd)  # after calculation, weights are assigned to every in-situ gages
        return fork_df

    @staticmethod
    def weight_normalization(ws):
        try:
            return [(m - min(ws)) / (max(ws) - min(ws)) for m in ws]
        except ZeroDivisionError:
            return [1] * len(ws)

    def elevation_weighting(self, fork_df):
        print(
            f'# -------------------------------1.3 Calculating elevation weights...-------------------------------#')
        fork_df['Wz'] = [0] * len(fork_df)
        fork_df['Wz'] = fork_df['Wz'].astype('object')
        for i in fork_df.iterrows():
            Wz = []
            for j in self.insitu_pcps.fork_df.iterrows():
                if j[1]['id'] != i[1]['id']:
                    delta_z = np.fabs(j[1]['dem0'] - i[1]['dem0'])
                    if delta_z <= self.MINIMUM_ELEVATION_DIFFERENCE:
                        Wz.append(1 / ((self.MINIMUM_ELEVATION_DIFFERENCE) ** self.ELEVATION_WEIGHTING_EXPONENT))
                    elif self.MAXIMUM_ELEVATION_DIFFERENCE >= delta_z > self.MINIMUM_ELEVATION_DIFFERENCE:
                        Wz.append(1 / ((delta_z) ** self.ELEVATION_WEIGHTING_EXPONENT))
                    else:
                        Wz.append(0)
                    # if Wz[-1] < self.MINIMUM_ELEVATION_WEIGHTING:
                    #     Wz[-1] = self.MINIMUM_ELEVATION_WEIGHTING
            fork_df.at[i[0], 'Wz'] = self.weight_normalization(
                Wz)  # after calculation, weights are assigned to every in-situ gages
        return fork_df

    def facet_weighting(self, fork_df):
        """
        this function has been simplified
        :param fork_df:
        :return:
        """
        LEESIDE = [5, 6, 7, 8]
        print(
            f'# -------------------------------1.4 Calculating facet weights...-------------------------------#')
        fork_df['Wf'] = [0] * len(fork_df)
        fork_df['Wf'] = fork_df['Wf'].astype('object')
        temp_df = fork_df
        for i in fork_df.iterrows():
            Wf = []
            for j in self.insitu_pcps.fork_df.iterrows():
                if j[1]['id'] != i[1]['id']:
                    # ---------------------------------------------------------------------------delta_f
                    delta_fs = []  # higher filter radius, higher distance considered
                    for dem in enumerate(self.dems):
                        delta_fs.append(math.fabs(j[1]['aspect' + str(dem[0])] - i[1]['aspect' + str(dem[0])]))
                    # ---------------------------------------------------------------------------B
                    # B = cal_different_facet(j[1]['raster_x'], i[1]['raster_x'], j[1]['raster_y'], i[1]['raster_y'],
                    #                         self.dems[-1].aspect)
                    # if delta_fs[-1] <= 2 and B <= 1:
                    # -------------------------------------------leeside/windside
                    if ((int(j[1]['aspect' + str(dem[0])]) in LEESIDE) and (
                            int(i[1]['aspect' + str(dem[0])]) in LEESIDE)) or (
                            (not int(j[1]['aspect' + str(dem[0])]) in LEESIDE) and
                            (not int(i[1]['aspect' + str(dem[0])]) in LEESIDE)):
                        windaspect = 1
                    else:
                        windaspect = 0

                    if windaspect == 0:
                        Wf.append(0.01)
                    elif delta_fs[-1] == 0 or delta_fs[-2] == 0:
                        Wf.append(1)
                    else:
                        Wf.append((2 / (delta_fs[-1] + delta_fs[-2])) ** self.FACET_WEIGHTING_EXPONENT)
                        # Wf.append(1 / (np.mean(delta_fs) + B) ** self.FACET_WEIGHTING_EXPONENT)
            temp_df.at[i[0], 'Wf'] = self.weight_normalization(Wf)
        return temp_df

    def effective_terrain_weighting(self, fork_df):
        print(
            f'# -------------------------------1.5 Calculating effective_terrain weights...-------------------------------#')
        fork_df['Wt'] = [0] * len(fork_df)
        fork_df['Wt'] = fork_df['Wt'].astype('object')

        for i in fork_df.iterrows():
            Wt = []
            for j in self.insitu_pcps.fork_df.iterrows():
                if j[1]['id'] != i[1]['id']:
                    if i[1]['effective_terrain_index'] == j[1]['effective_terrain_index'] == 1:
                        Wt.append(1)
                    else:
                        Wt.append(
                            1 / (100 * (
                                np.fabs(i[1]['effective_terrain_index'] - j[1]['effective_terrain_index'])) ** (
                                         0.5 * (1 - i[1]['effective_terrain_index']))))
            fork_df.at[i[0], 'Wt'] = self.weight_normalization(Wt)
        return fork_df

    def cal_effective_terrain_index(self):
        """
        this determine weather it is valley or mountain's top determined by the filtered DEM,
        the filter redius changed according to the topography of research area, for DVH, the
        width of valley typically 3000-10000m
        :return:
        """
        print('# -------------------------------calculating effective terrain index-------------------------------#')
        # topographic_index_blur = DEM(dem_array=self.dem0.gaussian_blur(10), dem_info=self.dem0.dem_info)
        step1_array = self.__distance_convolution_numba(distance=self.EFFECTIVE_INDEX_RADIUS_LARGE, mode='min',
                                                        data=self.dem0.dem_array)
        step2_smooth = self.__distance_convolution_numba(distance=self.EFFECTIVE_INDEX_RADIUS_SMALL, mode='mean',
                                                         data=step1_array)
        step3_difference = self.dem0.dem_array - step2_smooth
        # self.dem0.write_raster_file(input_array=step3_difference,im_geotrans = self.dem0.dem_info[4],im_proj = self.dem0.dem_info[5],path = r'./step3.tif')
        step4_difference_smooth_hc = self.__distance_convolution_numba(distance=self.EFFECTIVE_INDEX_RADIUS_SMALL,
                                                                       mode='mean',
                                                                       data=step3_difference)
        # self.dem0.write_raster_file(input_array=step4_difference_smooth_hc, im_geotrans=self.dem0.dem_info[4],
        #                             im_proj=self.dem0.dem_info[5], path=r'./effective_terrain_height1.tif')
        #  step4_difference_smooth_hc is the effective terrain height

        I3c = step4_difference_smooth_hc.copy()
        a = I3c[(I3c > self.TERRAIN_THRESHOLD_2D) & (I3c < self.TERRAIN_THRESHOLD_3D)]
        a = (a - self.TERRAIN_THRESHOLD_2D) / (self.TERRAIN_THRESHOLD_3D - self.TERRAIN_THRESHOLD_2D)
        I3c[step4_difference_smooth_hc >= self.TERRAIN_THRESHOLD_3D] = 1
        I3c[step4_difference_smooth_hc <= self.TERRAIN_THRESHOLD_2D] = 0
        I3c[(step4_difference_smooth_hc > self.TERRAIN_THRESHOLD_2D) & (
                step4_difference_smooth_hc < self.TERRAIN_THRESHOLD_3D)] = a

        self.dem0.write_raster_file(input_array=I3c, im_geotrans=self.dem0.dem_info[4],
                                    im_proj=self.dem0.dem_info[5], path=r'./I3c.tif')

        ha = self.__distance_convolution_numba(distance=self.EFFECTIVE_INDEX_RADIUS_SMALL, mode='mean',
                                               data=step4_difference_smooth_hc,
                                               kernel_type='IDW')
        I3a = ha.copy()
        a = I3a[(I3a > self.TERRAIN_THRESHOLD_2D) & (I3a < self.TERRAIN_THRESHOLD_3D)]
        a = (a - self.TERRAIN_THRESHOLD_2D) / (self.TERRAIN_THRESHOLD_3D - self.TERRAIN_THRESHOLD_2D)
        I3a[ha <= self.TERRAIN_THRESHOLD_2D] = 0
        I3a[ha >= self.TERRAIN_THRESHOLD_3D] = 1
        I3a[(ha > self.TERRAIN_THRESHOLD_2D) & (ha < self.TERRAIN_THRESHOLD_3D)] = a

        # self.dem0.write_raster_file(input_array=np.array((I3a, I3c)).max(axis=0), im_geotrans=self.dem0.dem_info[4],
        #                             im_proj=self.dem0.dem_info[5], path=r'./I3d.tif')
        I3d = np.array((I3a, I3c)).max(axis=0)
        return I3d

    def __distance_convolution_numba(self, data, distance, mode, kernel_type=None):
        distance_pixel = math.ceil((distance / 111319.55) / self.base_resolution)
        if distance_pixel == 0:
            distance_pixel = 1

        kernel_size = distance_pixel * 2 + 1
        padded_array = np.pad(data, distance_pixel, 'symmetric')
        n, m = padded_array.shape
        k = np.ones((distance_pixel * 2 + 1, distance_pixel * 2 + 1))
        if kernel_type == 'IDW':
            for kk in range(distance_pixel - 1):
                distances = (distance_pixel - kk) * self.base_resolution * 111319.55
                k[kk, :] = k[-(kk + 1), :] = k[:, -(kk + 1)] = k[:, kk] = distances
            k = k / np.sum(k)
        img_new = np.zeros(data.shape)
        return convolution_cuda(img_new, padded_array=padded_array, kernel_size=kernel_size, mode=mode, k=k)

    def __distance_convolution_pool(self, data, distance, mode, kernel_type=None):
        """
        this is a generalized convolution function
        it can conduct searching around a point, but not according to the pixels, but real distance
        :param data:
        :return:
        """
        distance_pixel = int((distance / 111319.55) / self.base_resolution)
        if distance_pixel == 0:
            distance_pixel = 1

        kernel_size = distance_pixel * 2 + 1
        padded_array = np.pad(data, distance_pixel, 'symmetric')
        n, m = padded_array.shape
        k = np.ones((distance_pixel * 2 + 1, distance_pixel * 2 + 1))
        if kernel_type == 'IDW':
            for kk in range(distance_pixel - 1):
                k[kk, :] = k[-(kk + 1), :] = k[:, -(kk + 1)] = k[:, kk] = (
                                                                                  distance_pixel - kk) * self.base_resolution * 111319.55

        img_new = np.zeros(data.shape)
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

    def cal_posi_nega_terrain(self):
        """
        this function is for identify positive-negative terrain, negative terrain related to valley, in the same valley, the affect of aspect can be omitted
        :return:
        """
        step1_smooth = self.__distance_convolution_numba(distance=self.POSI_NAGE_TERRAIN_RADIUS, mode='mean',
                                                         data=self.dem0.dem_array)
        step2_diff = self.dem0.dem_array - step1_smooth
        # step3_discrete = step2_diff.copy()
        # step3_discrete[step3_discrete >= 0] = 1
        # step3_discrete[step3_discrete < 0] = 0
        # self.dem0.write_raster_file(input_array=step2_diff, im_geotrans=self.dem0.dem_info[4],
        #                             im_proj=self.dem0.dem_info[5], path=r'./step2_diff.tif')
        return step2_diff

    def posi_nega_terrain_weighting(self, fork_df):
        print(
            f'# -------------------------------1.6 Calculating posi_nega_terrain weights...-------------------------------#')
        fork_df['Wp'] = [0] * len(fork_df)
        fork_df['Wp'] = fork_df['Wp'].astype('object')
        fork_df['Wpd'] = [0] * len(fork_df)
        fork_df['Wpd'] = fork_df['Wp'].astype('object')
        for i in fork_df.iterrows():
            Wp = []
            Wpd = []
            for j in self.insitu_pcps.fork_df.iterrows():
                if j[1]['id'] != i[1]['id']:
                    if i[1]['posi_nega_terrain_discrete'] < 0 and j[1]['posi_nega_terrain_discrete'] < 0:
                        Wp.append(0)
                    else:
                        Wp.append(1)
                    delta_pnt = math.fabs(i[1]['posi_nega_terrain_discrete'] - j[1]['posi_nega_terrain_discrete'])
                    try:
                        Wpd.append((1 / delta_pnt) ** self.POSI_NAGE_TERRAIN_WEIGHTING_EXPONENT)
                    except ZeroDivisionError:
                        Wpd.append(1)
            fork_df.at[i[0], 'Wp'] = Wp
            fork_df.at[i[0], 'Wpd'] = self.weight_normalization(Wpd)
        return fork_df


class Regression():
    def __init__(self, factors, weight_out_csv, mode='cross_validation'):
        self.factors = factors
        self.regression_dataset = factors.insitu_pcps
        if mode == 'cross_validation':
            self.interpolate_dataset = factors.insitu_pcps.fork_df
        else:
            self.interpolate_dataset = factors.all_forks

        # --------------if weights already generated, read it------
        weight_out_dir = r'.\03-PRISM-resultDIR' # 02-PRISM-processDIR
        weight_out_csv = os.path.join(weight_out_dir, weight_out_csv)
        if os.path.exists(weight_out_csv):
            weightdf = pandas.read_csv(weight_out_csv)
            for ww in ['Wc', 'Wd', 'Wz', 'Wf', 'Wt', 'Wp', 'Wpd']:
                weightdf[ww] = [eval(m.replace('nan', '0')) for m in weightdf[ww].values]
            self.interpolate_dataset = weightdf.drop(labels='Unnamed: 0', axis=1)
        else:
            # --------------2.0 bridge-----if the mode is cross_validation, then you do not need to bridge the vectors and rasters again, because it has done when define factors--------------------
            bridge_flag = False
            for m in self.interpolate_dataset.columns.tolist():
                if 'aspect' in m:
                    bridge_flag = True
                    break
            if not bridge_flag:
                self.interpolate_dataset = self.factors.bridge_vector_raster(self.interpolate_dataset)
            # ---------------2.1 cluster weighting---------------
            self.interpolate_dataset = self.factors.cluster_weighting(self.interpolate_dataset)
            # ---------------2.2 distance weighting--------------
            self.interpolate_dataset = self.factors.distance_weighting(self.interpolate_dataset)
            # ---------------2.3 elevation weighting--------------
            self.interpolate_dataset = self.factors.elevation_weighting(self.interpolate_dataset)
            # ---------------2.4 facet weighting--------------
            self.interpolate_dataset = self.factors.facet_weighting(self.interpolate_dataset)
            # ---------------2.5 effective terrain weighting--------------
            self.interpolate_dataset = self.factors.effective_terrain_weighting(self.interpolate_dataset)
            # ---------------2.6 positive-negative terrain weighting--------------
            self.interpolate_dataset = self.factors.posi_nega_terrain_weighting(self.interpolate_dataset)
            # ---------------WRITE WEIGHTS TO DIR-------------------------
            if not os.path.exists(weight_out_dir):
                os.makedirs(weight_out_dir)
            self.interpolate_dataset.to_csv(weight_out_csv)

    def __linear_slope(self, w, x, y):
        weights_50 = np.percentile(w, 50)
        x_50 = np.array(x)[w > weights_50]
        y_50 = np.array(y)[w > weights_50]
        w_50 = w[w > weights_50]
        x_hat = np.sum(np.multiply(w_50, x_50)) / np.sum(w_50)
        y_hat = np.sum(np.multiply(w_50, y_50)) / np.sum(w_50)
        lslp = np.sum(np.multiply(np.multiply(w_50, (x_50 - x_hat)), (y_50 - y_hat))) / np.sum(
            np.multiply(np.multiply(w_50, (x_50 - x_hat)), (x_50 - x_hat)))
        return x_hat, y_hat, lslp

    def __weighted_regression(self, w, x, y):
        linear_slope = 9
        exponent = 1
        while linear_slope < -0.1 or linear_slope > 0.05:
            x_hat, y_hat, linear_slope = self.__linear_slope(w ** exponent, x, y)
            exponent = exponent * 0.1
            if exponent < 1e-13:
                break
        linear_intercept = y_hat - np.dot(linear_slope, x_hat)
        # --------------------------------------------------------examine the linear regression
        p_f = 1
        # y_pred = linear_slope * x + linear_intercept
        # sst = np.sum((np.multiply(w, y) - y_hat) ** 2)
        # sse = np.sum((np.multiply(w, y_pred) - np.multiply(w, y)) ** 2)
        # p_f = 1 - (sse / sst)
        # -----------------------------------------------------------------------------------
        w_99 = w[w > np.percentile(w, 99)]
        y_hat_IDW = np.sum(np.multiply(w_99, np.array(y)[w > np.percentile(w, 99)])) / np.sum(w_99)
        return linear_slope, linear_intercept, p_f, y_hat_IDW

    def time_line(self, result_dir, part_flag='main_component'):
        total_forks = len(self.interpolate_dataset)

        # --------------------------per station

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        WIDW = False
        if part_flag == 'main_component':
            out_dir = os.path.join(result_dir, 'main_component')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            regression_df = self.regression_dataset.gage_df_smoothed.copy()
            dist_prop = 0.2

        elif part_flag == 'prop_component':
            out_dir = os.path.join(result_dir, 'prop_component')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            regression_df = self.regression_dataset.gage_df_perday_prop.copy()
            WIDW = True
            dist_prop = 0.2

        else:
            # if regression main component, then 1) outdir 2)regression df 3)weights(to nearest points in similar terrain)
            out_dir = os.path.join(result_dir, 'rest_component')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            regression_df = self.regression_dataset.gage_df_perday_frac.copy()
            WIDW = True
            dist_prop = 0.2

        for forkinfo in self.interpolate_dataset.iterrows():
            print(
                f'# -------------------------------2.1 regression for station {forkinfo[0] + 1}/{total_forks}...-------------------------------#')

            gene_file = os.path.join(out_dir, (str(forkinfo[1]['id']) + 'generated.txt'))
            if os.path.exists(gene_file):
                continue

            fork_i3d = forkinfo[1]['effective_terrain_index']

            fork_x = forkinfo[1]['dem1']

            w = forkinfo[1][['Wc', 'Wd', 'Wz', 'Wf', 'Wt', 'Wp', 'Wpd']].values

            if part_flag == 'main_component':
                combined_w = self.combined_weight(dist_prop, w, fork_i3d, mode='main')
            elif part_flag == 'prop_component' or part_flag == 'rest_component':
                combined_w = self.combined_weight(dist_prop, w, fork_i3d, mode='frac')
            else:
                print('we do not have this mode yet!')

            # --------------------------go into timeline
            fork_y = []
            real_y = []
            datetime = []

            # unknow point ids for regression
            ids = regression_df.columns.values.tolist()
            try:
                ids.remove(str(forkinfo[1]['id']))
            except ValueError:
                print(ids)
                print(forkinfo[1]['id'])
                pass

            for i in regression_df.iterrows():  # time id1 id2 id3 ...
                datetime.append(i[0])
                regression_x = \
                    self.regression_dataset.fork_df[~(self.regression_dataset.fork_df['id'] == forkinfo[1]['id'])][
                        'dem1'].values
                # real_y.append(i[1].loc[str(forkinfo[1]['id'])])
                regression_y = i[1].loc[ids].values
                # ---------------------first check the combined w
                fork_y.append(self.regression(WIDW, combined_w, fork_x, regression_x, regression_y))
            # pandas.DataFrame({'datetime': datetime, 'real_values': real_y, 'gene_values': fork_y}, ).to_csv(gene_file,
            #                                                                                                 index=False)
            pandas.DataFrame({'datetime': datetime, 'gene_values': fork_y}, ).to_csv(gene_file,
                                                                                     index=False)

    def regression(self, WIDW, combined_w, fork_x, regression_x, regression_y):
        linear_slope, linear_intercept, p_f, y_hat = self.__weighted_regression(combined_w, regression_x,
                                                                                regression_y)
        # ----------------------------------------post process----------------
        fork_y = linear_slope * fork_x + linear_intercept
        if linear_slope < -0.1 or linear_slope > 0.05:
            fork_y = y_hat
        if p_f < 0.6 and WIDW == True:
            # fork_y = np.sum(np.multiply(combined_w, regression_y)) / np.sum(combined_w)
            fork_y = y_hat
        return fork_y

    def combined_weight(self, dist_prop, w, fork_i3d, mode):
        s1 = np.power(
            (dist_prop * np.array(w[1]) ** 2 + (1 - dist_prop) * ((np.array(w[2]) ** 2) ** fork_i3d)), 0.5)
        if mode == 'main':
            combined_w = np.multiply(np.multiply(np.multiply(
                np.multiply(s1,
                            np.array(w[0])), ((np.array(w[3]) ** fork_i3d) ** np.array(w[5]))),
                np.array(w[4]) ** fork_i3d),
                np.array(w[6]) ** fork_i3d)
        else:
            combined_w = (s1 ** np.array(w[6])) ** fork_i3d
            combined_w[(np.array(w[1]) < 0.4) | (np.array(w[5]) == 1)] = 0
            if combined_w[combined_w != 0].size >= 3:
                mx = max(combined_w[combined_w != 0])
                mn = min(combined_w[combined_w != 0])
                combined_w[combined_w != 0] = (combined_w[combined_w != 0] - mn) / (mx - mn)
            else:
                combined_w[(np.array(w[1]) < 0.5)] = 0
        return combined_w


if __name__ == '__main__':
    import os

    f = FactorsCalculator(
        dem_filename=r'.\00_DEM\mosaic.tif',
        blur_steps=[4, 8, 16, 32, 64, 128],
        fork_file=r'.\00_ForksGages\pfork.txt', # r'.\00_ForksGages\pfork.txt',
        gage_dir=r'.\00_ForksGages', # r'.\00_ForksGages',
        gene_fork_file=r'.\00_InterpolatingPoints\HH_200_fishnet.txt',
        # gene_fork_file=r'.\00-shps\subs_centroids.shp',
        gene_gage_dir=None, influence_radius=50000, minimum_influence_radius=7000,
        elevation_weighting_exponent=1.0, minimum_elevation_difference=300.0,
        maximum_elevation_difference=1500,
        facet_weighting_exponent=2.0)
    r = Regression(factors=f, mode='generate',
                   weight_out_csv=r'weights_all_200_considerLWAspect.csv')  # 'weights_all_200_considerLWAspect.csv'
    r.time_line(result_dir=r'.\02-PRISM-Main-2000-200_fishnet_considerLWAspect') 
    # r.time_line(part_flag='prop_component', result_dir=r'./dis_fft_new')
    # r.time_line(part_flag='rest_component', result_dir=r'./dis_fft_new')
