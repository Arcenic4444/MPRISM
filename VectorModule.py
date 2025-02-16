import sys, os, glob
import pandas
from osgeo import gdal, ogr, osr
from PRISMpyUtil import *

class ClimateStations():
    def __init__(self, fork_file, gage_dir):
        self.gage_starttime = 0
        self.fork_file = fork_file
        self.gage_dir = gage_dir
        self.fork_df = self.__get_fork_df()
        self.gage_df = None
        self.gage_times_range = None

    def __get_fork_df(self):
        if self.fork_file[-3:] == 'txt':
            fork_df = pandas.read_csv(self.fork_file)
            fork_df = fork_df.rename(columns={'ID': 'id'})
            return fork_df

        elif self.fork_file[-3:] == 'shp':
            # make sure there are 'id' 'name' fields in the shapefile, and it's under WGS84
            gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
            gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
            ogr.RegisterAll()
            ds = ogr.Open(self.fork_file, 0)
            oLayer = ds.GetLayerByIndex(0)
            result_list = {}

            result_list['id'] = []
            result_list['NAME'] = []
            result_list['LONG'] = []
            result_list['LAT'] = []
            for i in range(0, oLayer.GetFeatureCount(0)):
                ofeature = oLayer.GetFeature(i)
                """ add '999','p_g_'to avoid duplicated names and ids"""
                result_list['id'].append('999'+ ofeature.GetFieldAsString("Subbasin"))
                result_list['NAME'].append('p_g_'+'999'+ (ofeature.GetFieldAsString("Subbasin")))
                result_list['LONG'].append(ofeature.GetGeometryRef().GetX())
                result_list['LAT'].append(ofeature.GetGeometryRef().GetY())
            ds.Destroy()
            del ds
            return pandas.DataFrame(result_list)

        else:
            raise Exception("Only accept files in shapefile or txt formats")

    def __get_gage_df(self):
        return 0

    def __get_gage_times_range(self):
        return 0


class PcpGagePnts(ClimateStations):
    def __init__(self, fork_file, gage_dir):
        super().__init__(fork_file, gage_dir)
        self.__get_gage_df()
        self.gage_times_range = None

    def __get_gage_df(self):
        """
        The input gauges may have uneven time distributions. Please ensure that all gauges used for analysis or interpolation fall within the same time range to maintain consistency and accuracy in the results.
        """
        print(
            f'# -------------------------------Reading gage files-------------------------------#')
        pcp_list = []
        pcp_smoo_list = []
        pcp_frac_list = []
        pcp_prop_list = []
        gage_timerange = pandas.date_range('1990-01-01', periods=1, freq='D')
        content_max = 0
        for pcpfile in glob.glob(os.path.join(self.gage_dir, 'p[0-9]*.txt')):
            gage_id = pcpfile.split('\\')[-1].split('.')[0][1:]
            if not int(gage_id) in self.fork_df['id'].values:
                raise Exception("Please check your input files to make sure the forks are in line with gauge files")
            else:
                with open(pcpfile) as pcp:
                    content = pcp.readlines()
                    yyyy = content[0].strip()[:4]
                    mm = content[0].strip()[4:6]
                    dd = content[0].strip()[-2:]
                    gage_df = pandas.DataFrame([float(m.strip()) for m in content[1:]],
                                               index=pandas.date_range('-'.join([yyyy, mm, dd]),
                                                                       periods=len(content) - 1, freq='D'),
                                               columns=[gage_id])
                    if len(content) > content_max:
                        content_max = len(content)
                        # print(content_max)
                        gage_starttime = '-'.join([yyyy, mm, dd])
                        gage_timerange = pandas.date_range(gage_starttime, periods=len(content) - 1, freq='D')
                        gage_df.reindex(gage_timerange)
                    # Calculate the climatology precipitation
                    gage_df_smoothed, gage_df_perday_prop, gage_df_perday_frac = fft_gages(gage_df.copy())
                    pcp_list.append(gage_df['2000-01-01':]) #.resample('MS').mean()
                    pcp_smoo_list.append(gage_df_smoothed) #.resample('MS').mean()
                    pcp_frac_list.append(gage_df_perday_frac['2000-01-01':]) #.resample('MS').mean()
                    pcp_prop_list.append(gage_df_perday_prop['2000-01-01':]) #.resample('MS').mean()
        self.gage_df = pandas.concat(pcp_list, axis=1)
        self.gage_df_smoothed = pandas.concat(pcp_smoo_list, axis=1)
        self.gage_df_perday_frac = pandas.concat(pcp_frac_list, axis=1)
        self.gage_df_perday_prop = pandas.concat(pcp_prop_list, axis=1)



class TmpGagePnts(ClimateStations):
    def __init__(self, fork_file, gage_dir):
        super().__init__(fork_file, gage_dir)

    """
    Ensure that the interpolated minimum temperature values are always less than the interpolated maximum temperature values. This ensures logical consistency and accuracy in the temperature data.
    """

    def __get_gage_df(self):
        tmx_list = {}
        tmn_list = {}
        for tmpfile in self.__gage_file:
            gage_id = tmpfile.split(r'\\')[-1].split('.')[0][1:]
            if not gage_id in self.fork_df['id']:
                raise Exception("Please check your input files to make sure the forks are in line with gauge files")
            else:
                with open(tmpfile) as tmp:
                    feng_tmp_value = [m.strip() for m in tmp.readlines()[1:]]
                    # feng_pcp_list[iid] = feng_pcp_value
                    tmx_list[gage_id] = [float(m.split(',')[0]) for m in tmp]
                    tmn_list[gage_id] = [float(m.split(',')[1]) for m in tmp]
        return 0


class PcpGenePnts(ClimateStations):
    def __init__(self, fork_file, gage_dir=None):
        super().__init__(fork_file, gage_dir)

    def write_gene_data(self):
        gene_fn = 'p_g'
        pass


# class VectorProcedures():
#     """
#     todo: unify generated pns format same as PcpGagePnts
#     """
#
#     def generate_new_forks(self, how, extent, output_dir, points_number=None, hspacing=None, vspacing=None):
#         """
#         :param how: 'random': generate interpolating stations randomly; 'grid': generate stations grid in the watershed
#         :return:
#         """
#         output_path = os.path.join(output_dir, 'generated_interpolate_points.shp')
#         crs = QgsCoordinateReferenceSystem("EPSG:2964")
#
#         if how == 'random':
#             parameter_dictionary = {
#                 'EXTENT': extent,
#                 'POINTS_NUMBER': points_number,
#                 'MIN_DISTANCE': 0,
#                 'TARGET_CRS': crs,
#                 'MAX_ATTEMPTS': 200,
#                 'OUTPUT': output_path}
#             processing.run("native:randompointsinextent", parameter_dictionary)
#
#         elif how == 'grid':
#             parameter_dictionary = {
#                 'TYPE': 0,  # 0 — Point
#                 'EXTENT': extent,
#                 'HSPACING': hspacing,
#                 'VSPACING': vspacing,
#                 'CRS': crs,
#                 'OUTPUT': output_path}
#             processing.run("native:native:creategrid", parameter_dictionary)
#
#         return output_path
#
#     def interpolate(self):
#         pass
#

if __name__ == '__main__':
    pcp = PcpGagePnts(r'.\00_ForksGages\pfork.txt',
                      r'.\00_ForksGages')
    print(pcp.gage_df_perday_frac,pcp.gage_df,pcp.gage_df_smoothed
          )
    # gene_pcp = PcpGenePnts(r'./test_gene.shp')
    # print(gene_pcp.gage_df_perday_frac)
    # print(pandas.concat([pcp.fork_df,gene_pcp.fork_df],ignore_index=True))
