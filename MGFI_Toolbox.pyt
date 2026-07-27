# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import unicode_literals
import arcpy
import numpy as np
import os
import io
import time
from arcpy.sa import *
from scipy import ndimage
from scipy.ndimage import convolve
from collections import deque

# ==============================================================================
#  Modified Geomorphic Flood Index (MGFI) Toolbox
#  Channel index modes: Channel_ASk (slope-angle) or Channel_GFI20 (hydraulic gradient, default).
#  Reference: Manfreda et al. (2017), Advances in Geosciences, 44, 9-19.
# ==============================================================================


class Toolbox(object):
    def __init__(self):
        self.label = "Modified Geomorphic Flood Index Toolbox"
        self.alias = "gfi"
        self.tools = [GeomorphicFloodArea]


class GeomorphicFloodArea(object):
    def __init__(self):
        self.label = "Modified Geomorphic Flood Index"
        self.description = (
            "Maps flood-prone areas using a geomorphology-based index, "
            "modified with rainfall, land cover, and soil runoff weighting (SCS-CN, Green-Ampt, ICL)."
        )
        self.canRunInBackground = False

    # --------------------------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------------------------
    def getParameterInfo(self):

        # -- A: INPUT LAYERS ---------------------------------------------------
        p_dem = arcpy.Parameter(
            displayName="DEM",
            name="dem", datatype="GPRasterLayer",
            parameterType="Required", direction="Input",
            category="A - Input Layers"
        )
        p_dem.description = (
            "DTM raster (not DSM) used to derive flow direction, slope, and height above nearest channel. "
            "Recommended: 30 m SRTM or DEMNAS."
        )
        p_landcover = arcpy.Parameter(
            displayName="Land Cover shapefile (field: CN)",
            name="shp_tutupan", datatype="GPFeatureLayer",
            parameterType="Optional", direction="Input",
            category="A - Input Layers"
        )
        p_landcover.description = (
            "Land cover shapefile with a CN field, used by SCS-CN and ICL runoff methods. "
            "Required if the runoff method is not conventional GFI."
        )
        p_soil = arcpy.Parameter(
            displayName="Soil shapefile (fields: INFILRATE, POROSITY, Sf, LOSS_RATE)",
            name="shp_tanah", datatype="GPFeatureLayer",
            parameterType="Optional", direction="Input",
            category="A - Input Layers"
        )
        p_soil.description = (
            "Soil shapefile with INFILRATE, POROSITY, Sf, and LOSS_RATE fields, used by Green-Ampt and ICL methods. "
            "Required if the runoff method is not conventional GFI."
        )
        p_rainfall = arcpy.Parameter(
            displayName="Design rainfall raster (mm)",
            name="p100", datatype="GPRasterLayer",
            parameterType="Optional", direction="Input",
            category="A - Input Layers"
        )
        p_rainfall.description = (
            "Design rainfall raster (mm), e.g. 100-year return period, used as input P for runoff ratio calculation. "
            "Required if the runoff method is not conventional GFI."
        )

        # -- B: METHOD OPTIONS -------------------------------------------------
        p_flow_coding = arcpy.Parameter(
            displayName="Flow direction encoding",
            name="flow_dir_coding", datatype="GPString",
            parameterType="Required", direction="Input",
            category="B - Method Options"
        )
        p_flow_coding.filter.type = "ValueList"
        p_flow_coding.filter.list = ["ESRI"]
        p_flow_coding.value = "ESRI"
        p_flow_coding.description = (
            "Flow direction encoding scheme. Only ESRI's standard D8 scheme is currently supported."
        )

        p_runoff = arcpy.Parameter(
            displayName="Runoff weighting method",
            name="runoff_method", datatype="GPString",
            parameterType="Required", direction="Input",
            category="B - Method Options"
        )
        p_runoff.filter.type = "ValueList"
        p_runoff.filter.list = [
            "GFI",
            "GFI - SCS CN",
            "GFI - Green Ampt",
            "GFI - Initial and Constant Loss",
            "Recommendation",
        ]
        p_runoff.value = "GFI"
        p_runoff.description = (
            "Runoff weighting method for flow accumulation: GFI (unweighted), SCS-CN, Green-Ampt, or "
            "Initial and Constant Loss. Recommendation runs all methods and reports the best."
        )

        # [6] Channel index method
        p_channel_mode = arcpy.Parameter(
            displayName="Channel index method",
            name="channel_mode", datatype="GPString",
            parameterType="Required", direction="Input",
            category="B - Method Options"
        )
        p_channel_mode.filter.type = "ValueList"
        p_channel_mode.filter.list = [
            "Channel_ASk",
            "Channel_GFI20",
        ]
        p_channel_mode.value = "Channel_GFI20"
        p_channel_mode.description = (
            "Horn 3x3 kernel channel index method: Channel_ASk (slope-angle, hilly terrain) or "
            "Channel_GFI20 (hydraulic gradient, flat terrain, default)."
        )

        p_ci_threshold = arcpy.Parameter(
            displayName="Channel Threshold",
            name="drain_thr", datatype="GPDouble",
            parameterType="Required", direction="Input",
            category="B - Method Options"
        )
        p_ci_threshold.value = 15000.0
        p_ci_threshold.description = (
            "Channel index threshold separating stream pixels from non-stream. Lower = denser network. Default 15000."
        )

        p_seed_mode = arcpy.Parameter(
            displayName="Channel delineation mode",
            name="seed_mode", datatype="GPString",
            parameterType="Required", direction="Input",
            category="B - Method Options"
        )
        p_seed_mode.filter.type = "ValueList"
        p_seed_mode.filter.list = [
            "Fixed (use value above)",
            "Auto-iterate (find best channel delineation)",
        ]
        p_seed_mode.value = "Fixed (use value above)"
        p_seed_mode.description = (
            "Fixed uses the Channel Threshold above; Auto-iterate tests several thresholds and picks the best ROC AUC "
            "(requires flood reference map and ROC calibration)."
        )

        p_n_exp = arcpy.Parameter(
            displayName="Hydraulic scaling exponent (n)",
            name="hyd_exp", datatype="GPDouble",
            parameterType="Required", direction="Input",
            category="B - Method Options"
        )
        p_n_exp.value = 0.3544
        p_n_exp.description = (
            "Hydraulic scaling exponent in the Leopold and Maddock equation hr = a * (Ar)^n. Default 0.3544."
        )

        p_v2_iter = arcpy.Parameter(
            displayName="Backwater effect iterations (maximum iteration)",
            name="v2_iter", datatype="GPLong",
            parameterType="Required", direction="Input",
            category="B - Method Options"
        )
        p_v2_iter.value = 6
        p_v2_iter.description = (
            "Maximum iterations for the GFI v2 backwater algorithm that propagates confluence influence upstream. Default 6."
        )

        # -- C: CALIBRATION OPTIONS --------------------------------------------
        p_manual_thr = arcpy.Parameter(
            displayName="Use manual GFI threshold",
            name="manual_thr", datatype="GPBoolean",
            parameterType="Required", direction="Input",
            category="C - Calibration Options"
        )
        p_manual_thr.value = True
        p_manual_thr.description = (
            "Enable to set the flood-prone GFI threshold manually instead of via ROC calibration."
        )

        p_gfi_thr = arcpy.Parameter(
            displayName="Manual GFI threshold value",
            name="gfi_thr", datatype="GPDouble",
            parameterType="Optional", direction="Input",
            category="C - Calibration Options"
        )
        p_gfi_thr.value = -0.53
        p_gfi_thr.description = (
            "GFI threshold separating flood-prone from non-flood-prone pixels. Active only if manual threshold is enabled."
        )

        p_calibrate = arcpy.Parameter(
            displayName="Calibrate threshold via ROC (requires flood reference map)",
            name="calibrate", datatype="GPBoolean",
            parameterType="Required", direction="Input",
            category="C - Calibration Options"
        )
        p_calibrate.value = False
        p_calibrate.description = (
            "Auto-determine the optimal GFI threshold from the ROC curve. Requires a flood reference map."
        )

        p_roc_steps = arcpy.Parameter(
            displayName="ROC calibration step size (normalised range [-1, 1])",
            name="roc_steps", datatype="GPDouble",
            parameterType="Required", direction="Input",
            category="C - Calibration Options"
        )
        p_roc_steps.value = 0.001
        p_roc_steps.description = (
            "Step size for the ROC threshold search over normalized GFI range [-1, 1]. Smaller = finer but slower."
        )

        p_sfm = arcpy.Parameter(
            displayName="Flood reference map - binary (0 = dry, 1 = flooded)",
            name="sfm", datatype="GPRasterLayer",
            parameterType="Optional", direction="Input",
            category="C - Calibration Options"
        )
        p_sfm.description = (
            "Binary flood raster (0/1) used as ground truth for ROC calibration and performance metrics."
        )

        p_flood_points = arcpy.Parameter(
            displayName="Flood event points shapefile (for point-accuracy check)",
            name="flood_points", datatype="GPFeatureLayer",
            parameterType="Optional", direction="Input",
            category="C - Calibration Options"
        )
        p_flood_points.description = (
            "Historical flood point shapefile used to compute point accuracy. Optional."
        )

        # -- D: OUTPUT OPTIONS -------------------------------------------------
        p_out_folder = arcpy.Parameter(
            displayName="Output folder",
            name="out_folder", datatype="DEFolder",
            parameterType="Required", direction="Input",
            category="D - Output Options"
        )
        p_out_folder.description = (
            "Destination folder for output rasters, reports, and CSV files."
        )

        p_out_prefix = arcpy.Parameter(
            displayName="Output file prefix",
            name="out_prefix", datatype="GPString",
            parameterType="Required", direction="Input",
            category="D - Output Options"
        )
        p_out_prefix.value = "GFI"
        p_out_prefix.description = (
            "File name prefix for all outputs, e.g. 'GFI' produces GFI_Channel.tif, GFI_v1.tif, etc."
        )

        p_save_intermediate = arcpy.Parameter(
            displayName="Save intermediate rasters (row/col index arrays)",
            name="create_intermediate", datatype="GPBoolean",
            parameterType="Optional", direction="Input",
            category="D - Output Options"
        )
        p_save_intermediate.value = False
        p_save_intermediate.description = (
            "Save intermediate rasters (channel row/col indices, runoff ratios) for debugging."
        )

        p_water_depth = arcpy.Parameter(
            displayName="Calculate Water Depth",
            name="water_depth", datatype="GPBoolean",
            parameterType="Optional", direction="Input",
            category="D - Output Options"
        )
        p_water_depth.value = False
        p_water_depth.description = (
            "Compute water depth (hr*a - H) and hazard class (Low/Medium/High). Requires calibration."
        )

        p_export_strahler = arcpy.Parameter(
            displayName="Export Strahler stream order raster",
            name="export_strahler", datatype="GPBoolean",
            parameterType="Optional", direction="Input",
            category="D - Output Options"
        )
        p_export_strahler.value = False
        p_export_strahler.description = (
            "Export the Strahler stream order raster for network visualization and validation."
        )

        p_export_csv = arcpy.Parameter(
            displayName="Export performance report CSV (requires flood reference map)",
            name="export_csv", datatype="GPBoolean",
            parameterType="Optional", direction="Input",
            category="D - Output Options"
        )
        p_export_csv.value = True
        p_export_csv.description = (
            "Export a CSV performance report (AUC, CSI, Kappa, F1, etc.). Requires ROC calibration."
        )

        p_export_channel_idx = arcpy.Parameter(
            displayName="Export channel index raster (diagnostic)",
            name="export_channel_idx", datatype="GPBoolean",
            parameterType="Optional", direction="Input",
            category="D - Output Options"
        )
        p_export_channel_idx.value = False
        p_export_channel_idx.description = (
            "Export the raw channel index raster (before thresholding) for diagnostics."
        )

        # Derived outputs
        p_out_gfi = arcpy.Parameter(
            displayName="MGFI base raster",
            name="out_gfi", datatype="DERasterDataset",
            parameterType="Derived", direction="Output"
        )
        p_out_flood = arcpy.Parameter(
            displayName="MGFI flood-prone raster (with backwater effect)",
            name="out_flood", datatype="DERasterDataset",
            parameterType="Derived", direction="Output"
        )

        # Parameter index reference:
        # [0]  dem               A - Input Layers
        # [1]  shp_tutupan       A - Input Layers
        # [2]  shp_tanah         A - Input Layers
        # [3]  p100              A - Input Layers
        # [4]  flow_dir_coding   B - Method Options
        # [5]  runoff_method     B - Method Options
        # [6]  channel_mode      B - Method Options
        # [7]  drain_thr         B - Method Options  (Channel Threshold)
        # [8]  seed_mode         B - Method Options  (Channel Delineation Mode)
        # [9]  hyd_exp           B - Method Options
        # [10] v2_iter           B - Method Options
        # [11] manual_thr        C - Calibration Options
        # [12] gfi_thr           C - Calibration Options
        # [13] calibrate         C - Calibration Options
        # [14] roc_steps         C - Calibration Options  (only used if calibrate = True)
        # [15] sfm               C - Calibration Options
        # [16] flood_points      C - Calibration Options
        # [17] out_folder        D - Output Options
        # [18] out_prefix        D - Output Options
        # [19] create_inter      D - Output Options
        # [20] water_depth       D - Output Options
        # [21] export_strahler   D - Output Options
        # [22] export_csv        D - Output Options
        # [23] export_channel_idx      D - Output Options  (export channel index raster)
        # [24] out_gfi           Derived
        # [25] out_flood         Derived

        return [
            p_dem, p_landcover, p_soil, p_rainfall,
            p_flow_coding, p_runoff,
            p_channel_mode, p_ci_threshold, p_seed_mode,
            p_n_exp, p_v2_iter,
            p_manual_thr, p_gfi_thr, p_calibrate, p_roc_steps, p_sfm, p_flood_points,
            p_out_folder, p_out_prefix, p_save_intermediate,
            p_water_depth, p_export_strahler, p_export_csv, p_export_channel_idx,
            p_out_gfi, p_out_flood,
        ]

    # --------------------------------------------------------------------------
    def isLicensed(self):
        return arcpy.CheckExtension("Spatial") == "Available"

    def updateParameters(self, parameters):
        runoff = parameters[5].valueAsText or "GFI"
        needs_hydro = runoff not in ["GFI"]
        for idx in [1, 2, 3]:
            parameters[idx].enabled = needs_hydro

        parameters[12].enabled = bool(parameters[11].value)

        do_calib = bool(parameters[13].value)
        parameters[14].enabled = do_calib
        parameters[15].enabled = do_calib
        parameters[16].enabled = do_calib
        parameters[22].enabled = do_calib

        if do_calib:
            parameters[11].value = False
            parameters[12].enabled = False

        if (parameters[8].valueAsText or "").startswith("Auto") and not do_calib:
            parameters[13].value = True

        # Auto-update default threshold hint when channel mode changes
        ch_mode = parameters[6].valueAsText or ""
        if not parameters[7].altered:
            parameters[7].value = 15000.0

    def updateMessages(self, parameters):
        runoff = parameters[5].valueAsText or "GFI"
        needs_hydro = runoff not in ["GFI"]

        if needs_hydro:
            for idx, label in [
                (1, "Land Cover shapefile"),
                (2, "Soil shapefile"),
                (3, "Design rainfall raster"),
            ]:
                if not parameters[idx].value:
                    parameters[idx].setErrorMessage(
                        "{} is required for the '{}' method.".format(label, runoff)
                    )

        if bool(parameters[13].value) and not parameters[15].value:
            parameters[15].setErrorMessage(
                "A flood reference map is required for ROC threshold calibration."
            )

        if (parameters[8].valueAsText or "").startswith("Auto") and not bool(parameters[13].value):
            parameters[8].setErrorMessage(
                "Auto-iterate requires 'Calibrate threshold via ROC' to be enabled "
                "and a flood reference map to be provided."
            )

    # ==========================================================================
    # MAIN EXECUTION
    # ==========================================================================
    def execute(self, parameters, messages):
        arcpy.CheckOutExtension("Spatial")
        arcpy.env.overwriteOutput = True

        # -- Read parameters ---------------------------------------------------
        dem_in        = parameters[0].valueAsText
        shp_landcover = parameters[1].valueAsText
        shp_soil      = parameters[2].valueAsText
        p100_raster   = parameters[3].valueAsText

        runoff_method  = parameters[5].valueAsText or "GFI"
        channel_mode   = parameters[6].valueAsText or "Channel_GFI20"
        use_ask   = channel_mode.startswith("Channel_ASk")
        use_GFI20  = channel_mode.startswith("Channel_GFI20")
        ci_threshold   = float(parameters[7].value)
        seed_mode_str  = parameters[8].valueAsText or "Fixed (use value above)"
        n_exp          = float(parameters[9].value)
        v2_max_iter    = int(parameters[10].value)

        use_manual_thr = bool(parameters[11].value)
        manual_thr_val = float(parameters[12].value) if parameters[12].value else -0.53
        do_calibrate   = bool(parameters[13].value)
        roc_steps      = float(parameters[14].value)
        sfm_raster     = parameters[15].valueAsText
        flood_points   = parameters[16].valueAsText

        out_folder        = parameters[17].valueAsText
        out_prefix        = (parameters[18].valueAsText or "GFI").strip()
        save_intermediate = bool(parameters[19].value)
        do_water_depth    = bool(parameters[20].value)
        do_strahler       = bool(parameters[21].value)
        do_export_csv     = bool(parameters[22].value)
        do_export_channel_idx   = bool(parameters[23].value)

        do_auto_ci = seed_mode_str.startswith("Auto") and do_calibrate

        # Candidate thresholds for auto-iteration (both channel modes share this scale)
        HORN_CANDIDATES = [50000, 30000, 15000, 8000, 5000, 2000, 1000]

        scratch = arcpy.env.scratchGDB

        # -- Variant selection -------------------------------------------------
        ALL_VARIANTS = ["CONVENTIONAL", "SCS_CN", "GREEN_AMPT", "ICL"]
        METHOD_MAP = {
            "GFI":                             ["CONVENTIONAL"],
            "GFI - SCS CN":                    ["SCS_CN"],
            "GFI - Green Ampt":                ["GREEN_AMPT"],
            "GFI - Initial and Constant Loss": ["ICL"],
            "Recommendation":                  ALL_VARIANTS,
        }
        variant_list   = METHOD_MAP.get(runoff_method, ["CONVENTIONAL"])
        single_variant = len(variant_list) == 1

        TAG = {
            "CONVENTIONAL": "KONV",
            "SCS_CN":       "SCS_CN",
            "GREEN_AMPT":   "GA",
            "ICL":          "ICL",
        }
        LABEL = {
            "CONVENTIONAL": "GFI",
            "SCS_CN":       "GFI - SCS CN",
            "GREEN_AMPT":   "GFI - Green Ampt",
            "ICL":          "GFI - Initial and Constant Loss",
        }

        # -- Helpers -----------------------------------------------------------
        def make_path(suffix, variant="", version=""):
            ver_tag = "_v{}".format(version) if version else ""
            if single_variant or not variant:
                # Single variant: prefix + suffix directly (e.g. GFI_Channel.tif)
                return os.path.join(out_folder, "{}{}{}.tif".format(out_prefix, suffix, ver_tag))
            # Multi variant: prefix + _TAG + suffix (e.g. GFI_GA_Channel.tif)
            return os.path.join(
                out_folder, "{}_{}{}{}.tif".format(out_prefix, TAG[variant], suffix, ver_tag)
            )

        def read_raster(path):
            # Matches Colab's read_raster_full(): read at float64 internally
            # (safe intermediate for the NoData comparison), then store as
            # float32 with NoData -> NaN, same as the Colab pipeline.
            r = arcpy.Raster(path)
            ndv = r.noDataValue if r.noDataValue is not None else -9999
            arr = arcpy.RasterToNumPyArray(path, nodata_to_value=ndv).astype(np.float64)
            arr[arr == ndv]  = np.nan
            arr[arr < -1e10] = np.nan
            return arr.astype(np.float32)

        # -- Spatial reference -------------------------------------------------
        ref_raster    = arcpy.Raster(dem_in)
        cell_size     = ref_raster.meanCellWidth
        lower_left    = arcpy.Point(ref_raster.extent.XMin, ref_raster.extent.YMin)
        spatial_ref   = ref_raster.spatialReference
        cell_size_str = str(int(round(cell_size)))
        cell_area     = cell_size * cell_size

        def save_raster_float(arr, path):
            clean = np.where(np.isnan(arr.astype(np.float64)), -9999.0, arr).astype(np.float32)
            r = arcpy.NumPyArrayToRaster(clean, lower_left, cell_size, cell_size, -9999.0)
            r.save(path)
            arcpy.DefineProjection_management(path, spatial_ref)

        def save_raster_int(arr, path, nodata=-9999):
            r = arcpy.NumPyArrayToRaster(
                arr.astype(np.int16), lower_left, cell_size, cell_size, nodata
            )
            r.save(path)
            arcpy.DefineProjection_management(path, spatial_ref)

        arcpy.env.snapRaster = dem_in
        arcpy.env.extent     = dem_in
        arcpy.env.cellSize   = cell_size_str
        arcpy.env.outputCoordinateSystem = dem_in

        # ======================================================================
        # STEP 1 - DEM preprocessing: Fill -> Flow Direction -> Slope
        # ======================================================================
        messages.addMessage("=" * 60)
        messages.addMessage("STEP 1: DEM Preprocessing")

        fill_dem   = os.path.join(scratch, "gfa_fill_dem")
        flow_dir   = os.path.join(scratch, "gfa_flowdir")
        slope_path = os.path.join(scratch, "gfa_slope_deg")

        # No z_limit -> fill every sink, same as the manual reference run
        Fill(dem_in).save(fill_dem)
        messages.addMessage("  Filled DEM    : done")

        # Explicit Force Flow / D8 so this matches the GUI default used
        # for the manual reference run instead of relying on tool defaults
        # that can silently differ between a scripted call and the ArcMap GUI.
        flowdir_raw = FlowDirection(
            in_surface_raster=fill_dem,
            force_flow="NORMAL",
            flow_direction_type="D8",
        )
        flowdir_raw.save(flow_dir)
        messages.addMessage("  Flow Direction: done")

        Slope(fill_dem, "DEGREE").save(slope_path)
        messages.addMessage("  Slope (deg)   : done")

        # Permanent exports so Colab can consume the EXACT SAME topology
        # this toolbox run produced, instead of a separately/manually
        # generated Fill/FlowDir/FlowAcc raster that may not match
        # byte-for-byte (different Snap Raster / Extent / Cell Size /
        # tool parameters used in a manual ArcGIS run can shift results
        # at pit/flat pixels even from the identical input DEM).
        fill_dem_perm   = os.path.join(out_folder, "{}_FILL_DEM.tif".format(out_prefix))
        flow_dir_perm   = os.path.join(out_folder, "{}_FLOWDIR.tif".format(out_prefix))
        slope_deg_perm  = os.path.join(out_folder, "{}_SLOPE_DEG.tif".format(out_prefix))
        arcpy.CopyRaster_management(fill_dem, fill_dem_perm)
        arcpy.CopyRaster_management(flow_dir, flow_dir_perm)
        arcpy.CopyRaster_management(slope_path, slope_deg_perm)
        messages.addMessage("  Exported (permanent):")
        messages.addMessage("    Filled DEM     -> {}".format(fill_dem_perm))
        messages.addMessage("    Flow Direction -> {}".format(flow_dir_perm))
        messages.addMessage("    Slope (deg)    -> {}".format(slope_deg_perm))

        arcpy.env.snapRaster = fill_dem
        arcpy.env.extent     = fill_dem

        dem_arr = read_raster(fill_dem)
        nrows, ncols = dem_arr.shape

        flow_dir_arr = arcpy.RasterToNumPyArray(
            flow_dir, nodata_to_value=0
        ).astype(np.int32)

        slope_deg_arr = read_raster(slope_path)

        # tan(slope), used as a fallback before Horn kernel gradients are computed below
        slope_tan_arr = np.tan(np.radians(
            np.clip(np.where(np.isnan(slope_deg_arr), 0.0, slope_deg_arr), 0.0, 89.0)
        ))
        slope_tan_arr = np.where(slope_tan_arr < 1e-6, 1e-6, slope_tan_arr)

        # ----------------------------------------------------------------
        # Horn 3x3 slope, computed for both channel index modes.
        # slope_rad_arr : arctan(grad)  [radians]   -> used by Channel_ASk
        # slope_tan_arr : Horn gradient (= tan(slope_rad)) -> used by Channel_GFI20
        # ----------------------------------------------------------------
        EPS_ASK = 0.0001   # numerical floor for slope_rad in Channel_ASk
        messages.addMessage("  Computing Horn 3x3 slope (rad + tan)...")
        dem_for_horn = dem_arr.copy()
        dem_nan_mask = np.isnan(dem_for_horn)
        dem_for_horn[dem_nan_mask] = float(np.nanmean(dem_for_horn))

        cs = abs(cell_size)
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64) / (8.0 * cs)
        ky = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]], np.float64) / (8.0 * cs)

        dzdx = convolve(dem_for_horn, kx, mode="nearest")
        dzdy = convolve(dem_for_horn, ky, mode="nearest")
        _grad = np.sqrt(dzdx**2 + dzdy**2)

        slope_rad_arr = np.arctan(_grad).astype(np.float32)
        slope_rad_arr[dem_nan_mask] = np.nan

        # Overwrite slope_tan_arr with Horn-consistent values
        slope_tan_arr = np.where(_grad < 1e-6, 1e-6, _grad).astype(np.float32)
        slope_tan_arr[dem_nan_mask] = np.nan

        rad_min = float(np.nanmin(slope_rad_arr))
        rad_max = float(np.nanmax(slope_rad_arr))
        messages.addMessage(
            "  slope_rad: [{:.6f}, {:.6f}] rad  slope_tan: [{:.6f}, {:.6f}]".format(
                rad_min, rad_max,
                float(np.nanmin(slope_tan_arr)), float(np.nanmax(slope_tan_arr)))
        )

        D8_MOVES = {
            1:   (0,  1),
            2:   (1,  1),
            4:   (1,  0),
            8:   (1, -1),
            16:  (0, -1),
            32:  (-1, -1),
            64:  (-1, 0),
            128: (-1, 1),
        }

        messages.addMessage("  Grid size: {} cols x {} rows".format(ncols, nrows))

        def validate_inputs(dem, fa, fdir, flood_map=None):
            issues = []
            if np.all(np.isnan(dem)):
                issues.append("DEM is entirely NaN.")
            elif np.nanmax(dem) == np.nanmin(dem):
                issues.append("DEM is flat (no elevation variation).")
            if np.nanmax(fa) <= 0:
                issues.append("Flow Accumulation has no positive values.")
            if len(fdir[fdir != 0]) == 0:
                issues.append("Flow Direction raster has no valid values.")
            if flood_map is not None:
                unique_vals = np.unique(flood_map[~np.isnan(flood_map)])
                if not np.all(np.isin(unique_vals, [0, 1])):
                    issues.append("Flood reference map must be binary (0/1).")
                if np.all(flood_map == 0) or np.all(np.isnan(flood_map)):
                    issues.append("Flood reference map contains no flooded pixels.")
            if issues:
                for msg in issues:
                    messages.addMessage("  [VALIDATION] " + msg)
                return False
            messages.addMessage("  [VALIDATION] All inputs passed.")
            return True

        # ======================================================================
        # STEP 2 - Runoff ratios
        # ======================================================================
        needs_hydro = any(v != "CONVENTIONAL" for v in variant_list)
        rf_scs = rf_ga = rf_icl = None

        if needs_hydro:
            messages.addMessage("=" * 60)
            messages.addMessage("STEP 2: Computing runoff ratios (Pe/P)")

            cn_path  = os.path.join(scratch, "gfa_cn")
            ke_path  = os.path.join(scratch, "gfa_ke")
            por_path = os.path.join(scratch, "gfa_por")
            sf_path  = os.path.join(scratch, "gfa_sf")
            fc_path  = os.path.join(scratch, "gfa_fc")

            arcpy.PolygonToRaster_conversion(shp_landcover, "CN",        cn_path,  "CELL_CENTER", "NONE", cell_size_str)
            arcpy.PolygonToRaster_conversion(shp_soil,      "INFILRATE", ke_path,  "CELL_CENTER", "NONE", cell_size_str)
            arcpy.PolygonToRaster_conversion(shp_soil,      "POROSITY",   por_path, "CELL_CENTER", "NONE", cell_size_str)
            arcpy.PolygonToRaster_conversion(shp_soil,      "Sf",         sf_path,  "CELL_CENTER", "NONE", cell_size_str)
            arcpy.PolygonToRaster_conversion(shp_soil,      "LOSS_RATE",  fc_path,  "CELL_CENTER", "NONE", cell_size_str)

            P    = read_raster(p100_raster)
            CN   = read_raster(cn_path)
            Ksat = read_raster(ke_path)
            Por  = read_raster(por_path)
            Sf   = read_raster(sf_path)
            fc   = read_raster(fc_path)

            S_curve = (25400.0 - 254.0 * CN) / CN
            Ia_scs  = 0.2 * S_curve
            Pe_scs  = np.where(P > Ia_scs,
                               (P - Ia_scs) ** 2 / (P - Ia_scs + S_curve),
                               0.0)
            rf_scs  = np.clip(np.where(P > 0, Pe_scs / P, np.nan), 0.0, 1.0)
            rf_scs[np.isnan(P) | np.isnan(CN)] = np.nan

            # SCS Type II 24-hr storm, cumulative fraction of total depth (25 points, t=0..24h)
            _type2_cum = np.array([
                0.000, 0.011, 0.022, 0.035, 0.048, 0.063, 0.080, 0.098, 0.120,
                0.147, 0.181, 0.235, 0.663, 0.772, 0.820, 0.850, 0.880, 0.898,
                0.916, 0.930, 0.944, 0.958, 0.971, 0.986, 1.000
            ])
            hourly_frac = np.diff(_type2_cum)   # 24 hourly increments, sum = 1.0

            theta_Sf = Por * Sf
            Ke       = Ksat   # used directly, no Ksat->Ke transform

            def run_ga_hourly(Ke_arr):
                F_prev = np.zeros_like(P)
                ponded = np.zeros_like(P, dtype=bool)
                for t in range(24):
                    r_t = P * hourly_frac[t]
                    cap = Ke_arr * (1.0 + theta_Sf / np.maximum(F_prev, 1e-6))
                    ponded = ponded | (r_t > cap)

                    F_imp = F_prev + Ke_arr
                    for _ in range(10):
                        num = np.maximum(F_prev + theta_Sf, 1e-9)
                        den = np.maximum(F_imp + theta_Sf, 1e-9)
                        g   = F_imp - F_prev - Ke_arr - theta_Sf * np.log(den / num)
                        gp  = 1.0 - theta_Sf / den
                        gp  = np.where(np.abs(gp) < 1e-9, 1e-9, gp)
                        F_imp = np.maximum(F_imp - g / gp, F_prev)

                    F_simple = F_prev + r_t
                    F_prev = np.where(ponded, F_imp, F_simple)
                return F_prev

            # Single forward-run (no calibration): F24 = F(t=24)
            F24 = run_ga_hourly(Ke)

            rf_ga  = np.clip(np.where(P > 0, 1.0 - F24 / P, np.nan), 0.0, 1.0)
            rf_ga[np.isnan(P) | np.isnan(Ksat) | np.isnan(CN) | np.isnan(Por) | np.isnan(Sf)] = np.nan

            rf_icl = np.clip(
                np.where(P > 0, 1.0 - (Ia_scs + fc * 24.0) / P, np.nan), 0.0, 1.0
            )
            rf_icl[np.isnan(P) | np.isnan(CN) | np.isnan(fc)] = np.nan

            if save_intermediate:
                save_raster_float(rf_scs, make_path("_Rf_SCSCN"))
                save_raster_float(rf_ga,  make_path("_Rf_GA"))
                save_raster_float(rf_icl, make_path("_Rf_ICL"))

            messages.addMessage("  Runoff ratios computed.")

        # ======================================================================
        # STEP 3 - Weighted Flow Accumulation
        # ======================================================================
        messages.addMessage("=" * 60)
        messages.addMessage("STEP 3: Weighted Flow Accumulation")

        fdir_raster = arcpy.Raster(flow_dir)
        RF_MAP = {
            "CONVENTIONAL": None,
            "SCS_CN":       rf_scs,
            "GREEN_AMPT":   rf_ga,
            "ICL":          rf_icl,
        }

        fa_path_map  = {}
        fa_array_map = {}

        for variant in variant_list:
            rf_arr  = RF_MAP[variant]
            fa_path = os.path.join(scratch, "gfa_fa_{}".format(TAG[variant]))

            if rf_arr is None:
                FlowAccumulation(fdir_raster, data_type="INTEGER").save(fa_path)
                messages.addMessage("  FlowAcc [{}]: unweighted (INTEGER)".format(LABEL[variant]))
            else:
                weight_arr = rf_arr.copy()
                weight_arr[np.isnan(weight_arr)] = 0.0
                weight_raster_path = os.path.join(scratch, "gfa_weight_{}".format(TAG[variant]))
                weight_raster = arcpy.NumPyArrayToRaster(weight_arr, lower_left, cell_size, cell_size, 0.0)
                weight_raster.save(weight_raster_path)
                arcpy.DefineProjection_management(weight_raster_path, spatial_ref)
                FlowAccumulation(fdir_raster, weight_raster_path, data_type="INTEGER").save(fa_path)
                messages.addMessage("  FlowAcc [{}]: weighted by Pe/P (INTEGER)".format(LABEL[variant]))

            fa_path_map[variant] = fa_path

            # Permanent export: lets Colab read the exact same FlowAcc
            # this toolbox run produced for this variant, instead of a
            # separately-generated static raster.
            fa_perm_path = os.path.join(
                out_folder, "{}_FLOWACC_{}.tif".format(out_prefix, TAG[variant])
            )
            arcpy.CopyRaster_management(fa_path, fa_perm_path)

            # Read FlowAcc the same way Colab's read_raster_full() does:
            # float32 precision, and true NoData pixels excluded as NaN
            # (not silently treated as a valid flow accumulation of 0).
            fa_raster_obj = arcpy.Raster(fa_path)
            fa_ndv = fa_raster_obj.noDataValue if fa_raster_obj.noDataValue is not None else -9999
            fa_arr_raw = arcpy.RasterToNumPyArray(
                fa_path, nodata_to_value=fa_ndv
            ).astype(np.float32)
            fa_arr_raw[fa_arr_raw == fa_ndv] = np.nan
            fa_array_map[variant] = fa_arr_raw
            messages.addMessage(
                "    FlowAcc [{}] range=[{:.1f}, {:.1f}]  -> {}".format(
                    LABEL[variant], float(np.nanmin(fa_arr_raw)),
                    float(np.nanmax(fa_arr_raw)), fa_perm_path)
            )

        messages.addMessage("  Weighted flow accumulation done.")

        # ======================================================================
        # CORE FUNCTIONS
        # ======================================================================

        # ----------------------------------------------------------------------
        # Horn-based channel index computation.
        #   Channel_ASk   = FA x cs2 x (slope_rad + eps)^1.7   - landform/angle sensitive
        #   Channel_GFI20 = FA x cs2 x tan(slope_Horn)^1.7      - hydraulic gradient
        #
        # The selected index is used ONLY as the channel seed criterion.
        # After the channel mask is formed, all downstream GFI computations
        # (hr, H, backwater) use the variant-specific fa_arr.
        # ----------------------------------------------------------------------
        def compute_channel_indices(fa_arr):
            """
            Compute the Channel_ASk and Channel_GFI20 index arrays for a given
            flow accumulation array.

                Channel_ASk   = FA x cs2 x (slope_rad + eps)^1.7
                Channel_GFI20 = FA x cs2 x tan(slope)^1.7

            Both share the same FA (variant-specific), so runoff weighting
            propagates into channel delineation.

            Returns: (ask_arr, gfi20_arr), both float32
            """
            cs2 = cell_area  # cell_size^2

            # Guard NaN
            FA = np.where(np.isnan(fa_arr) | (fa_arr < 0), 0.0, fa_arr)
            G  = np.where(np.isnan(slope_rad_arr), 0.0, slope_rad_arr)  # radian
            T  = slope_tan_arr  # already clipped to >= 1e-6

            ask_arr   = (FA * cs2 * np.power(G + EPS_ASK, 1.7)).astype(np.float32)
            gfi20_arr = (FA * cs2 * np.power(T, 1.7)).astype(np.float32)

            # Diagnostics
            for name, idx in [("Channel_ASk", ask_arr), ("Channel_GFI20", gfi20_arr)]:
                fin = idx[np.isfinite(idx) & (idx > 0)]
                if len(fin) > 0:
                    messages.addMessage(
                        "    {}: [{:.2e}, {:.2e}]  px>thr: {:,}".format(
                            name, float(fin.min()), float(fin.max()),
                            int((idx > ci_threshold).sum())
                        )
                    )
            return ask_arr, gfi20_arr

        def build_channel_network(fa_arr, ci_thr):
            """
            Build binary channel mask.

              Channel_ASk:   seed = Channel_ASk index > ci_thr
              Channel_GFI20: seed = Channel_GFI20 index > ci_thr

            After seeding, each seed pixel is traced downstream until it
            hits an existing channel pixel or exits the grid.
            For Channel_ASk, tracing also stops once the pixel elevation
            reaches the Channel Threshold.

            Returns: channel int8 array, ask_arr, gfi20_arr
            """
            ask_arr, gfi20_arr = compute_channel_indices(fa_arr)
            if use_ask:
                seed_index = ask_arr
                mode_label = "Channel_ASk"
            else:  # Channel_GFI20 (default)
                seed_index = gfi20_arr
                mode_label = "Channel_GFI20"

            seed_mask = seed_index > ci_thr
            n_seeds   = int(seed_mask.sum())
            messages.addMessage(
                "    [{}] threshold={:.2e}  seed pixels={:,}".format(
                    mode_label, ci_thr, n_seeds)
            )

            channel   = np.zeros((nrows, ncols), dtype=np.int8)
            max_steps = nrows + ncols

            for r0, c0 in zip(*np.where(seed_mask)):
                r, c, steps = int(r0), int(c0), 0
                if use_ask:
                    # Channel_ASk: tracing also stops once elevation reaches
                    # the Channel Threshold, or near the grid edge.
                    while (dem_arr[r, c] < ci_thr and 0 < r < nrows - 2
                           and 0 < c < ncols - 2 and steps < max_steps):
                        if channel[r, c] == 1 and (r != r0 or c != c0):
                            break
                        channel[r, c] = 1
                        fd = int(flow_dir_arr[r, c])
                        if fd not in D8_MOVES:
                            break
                        dr, dc = D8_MOVES[fd]
                        r += dr
                        c += dc
                        steps += 1
                else:
                    while 0 <= r < nrows and 0 <= c < ncols and steps < max_steps:
                        if channel[r, c] == 1 and (r != r0 or c != c0):
                            break
                        channel[r, c] = 1
                        fd = int(flow_dir_arr[r, c])
                        if fd not in D8_MOVES:
                            break
                        dr, dc = D8_MOVES[fd]
                        r += dr
                        c += dc
                        steps += 1

            channel[np.isnan(dem_arr)] = 0
            messages.addMessage(
                "    Channel pixels: {:,}".format(int(channel.sum()))
            )
            return channel, ask_arr, gfi20_arr

        def compute_strahler(channel):
            ch_rows, ch_cols = np.where(channel == 1)

            ds_r     = np.full((nrows, ncols), -1, dtype=np.int32)
            ds_c     = np.full((nrows, ncols), -1, dtype=np.int32)
            in_count = np.zeros((nrows, ncols), dtype=np.int16)

            for r, c in zip(ch_rows, ch_cols):
                fd = int(flow_dir_arr[r, c])
                if fd not in D8_MOVES:
                    continue
                dr, dc = D8_MOVES[fd]
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols and channel[nr, nc] == 1:
                    ds_r[r, c]    = nr
                    ds_c[r, c]    = nc
                    in_count[nr, nc] += 1

            strahler      = np.zeros((nrows, ncols), dtype=np.int16)
            max_order_in  = np.zeros((nrows, ncols), dtype=np.int16)
            max2_order_in = np.zeros((nrows, ncols), dtype=np.int16)
            pending       = in_count.copy()

            queue = deque()
            for r, c in zip(ch_rows, ch_cols):
                if in_count[r, c] == 0:
                    strahler[r, c] = 1
                    queue.append((r, c))

            while queue:
                r, c = queue.popleft()
                nr, nc = int(ds_r[r, c]), int(ds_c[r, c])
                if nr == -1:
                    continue
                s = int(strahler[r, c])
                if s > max_order_in[nr, nc]:
                    max2_order_in[nr, nc] = max_order_in[nr, nc]
                    max_order_in[nr, nc]  = s
                elif s > max2_order_in[nr, nc]:
                    max2_order_in[nr, nc] = s
                pending[nr, nc] -= 1
                if pending[nr, nc] == 0:
                    m1 = int(max_order_in[nr, nc])
                    m2 = int(max2_order_in[nr, nc])
                    strahler[nr, nc] = m1 + 1 if m1 == m2 else m1
                    queue.append((nr, nc))

            messages.addMessage(
                "    Max Strahler order: {}".format(int(np.max(strahler)))
            )
            return strahler

        def map_hillslope_to_channel(channel):
            ds_r = np.full((nrows, ncols), -1, dtype=np.int32)
            ds_c = np.full((nrows, ncols), -1, dtype=np.int32)
            for fd_val, (dr, dc) in D8_MOVES.items():
                mask = (flow_dir_arr == fd_val)
                src_r, src_c = np.where(mask)
                tgt_r = src_r + dr
                tgt_c = src_c + dc
                valid = (tgt_r >= 0) & (tgt_r < nrows) & (tgt_c >= 0) & (tgt_c < ncols)
                ds_r[src_r[valid], src_c[valid]] = tgt_r[valid]
                ds_c[src_r[valid], src_c[valid]] = tgt_c[valid]

            ROW_ch  = np.full((nrows, ncols), np.nan, dtype=np.float32)
            COL_ch  = np.full((nrows, ncols), np.nan, dtype=np.float32)
            ch_mask = channel == 1
            rr, cc  = np.where(ch_mask)
            ROW_ch[rr, cc] = rr.astype(np.float32)
            COL_ch[rr, cc] = cc.astype(np.float32)

            valid_hillslope = ~np.isnan(dem_arr) & ~ch_mask
            max_iter = nrows + ncols

            for _ in range(max_iter):
                unresolved = valid_hillslope & np.isnan(ROW_ch)
                if not unresolved.any():
                    break
                ur, uc   = np.where(unresolved)
                nr       = ds_r[ur, uc]
                nc       = ds_c[ur, uc]
                has_ds   = (nr >= 0)
                nr_safe  = np.where(has_ds, nr, 0)
                nc_safe  = np.where(has_ds, nc, 0)
                ds_known = ~np.isnan(ROW_ch[nr_safe, nc_safe])
                can_upd  = has_ds & ds_known
                if not can_upd.any():
                    break
                ROW_ch[ur[can_upd], uc[can_upd]] = ROW_ch[nr[can_upd], nc[can_upd]]
                COL_ch[ur[can_upd], uc[can_upd]] = COL_ch[nr[can_upd], nc[can_upd]]

            messages.addMessage(
                "    Hillslope pixels mapped: {:,}".format(int((~np.isnan(ROW_ch)).sum()))
            )
            return ROW_ch, COL_ch

        def map_channel_to_confluence(channel, strahler):
            ds_r = np.full((nrows, ncols), -1, dtype=np.int32)
            ds_c = np.full((nrows, ncols), -1, dtype=np.int32)
            for fd_val, (dr, dc) in D8_MOVES.items():
                mask = (flow_dir_arr == fd_val)
                src_r, src_c = np.where(mask)
                tgt_r = src_r + dr
                tgt_c = src_c + dc
                valid = (tgt_r >= 0) & (tgt_r < nrows) & (tgt_c >= 0) & (tgt_c < ncols)
                ds_r[src_r[valid], src_c[valid]] = tgt_r[valid]
                ds_c[src_r[valid], src_c[valid]] = tgt_c[valid]

            max_order = int(np.max(strahler)) if strahler.max() > 0 else 1
            ROW_conf  = np.full((nrows, ncols), np.nan, dtype=np.float32)
            COL_conf  = np.full((nrows, ncols), np.nan, dtype=np.float32)

            ch_r, ch_c = np.where(channel == 1)
            has_ds  = ds_r[ch_r, ch_c] >= 0
            nr_all  = np.where(has_ds, ds_r[ch_r, ch_c], 0)
            nc_all  = np.where(has_ds, ds_c[ch_r, ch_c], 0)
            s_self  = strahler[ch_r, ch_c]
            s_next  = strahler[nr_all, nc_all]
            is_conf = has_ds & (s_next > s_self) & (s_self < max_order)
            ROW_conf[ch_r[is_conf], ch_c[is_conf]] = nr_all[is_conf].astype(np.float32)
            COL_conf[ch_r[is_conf], ch_c[is_conf]] = nc_all[is_conf].astype(np.float32)

            max_iter = nrows + ncols
            for _ in range(max_iter):
                unresolved = (
                    (channel == 1) & np.isnan(ROW_conf) &
                    (strahler < max_order) & (strahler > 0)
                )
                if not unresolved.any():
                    break
                ur, uc  = np.where(unresolved)
                nr      = ds_r[ur, uc]
                nc      = ds_c[ur, uc]
                has_ds2   = (nr >= 0)
                nr_safe   = np.where(has_ds2, nr, 0)
                nc_safe   = np.where(has_ds2, nc, 0)
                resolved  = ~np.isnan(ROW_conf[nr_safe, nc_safe])
                same_ord  = strahler[nr_safe, nc_safe] == strahler[ur, uc]
                can_upd   = has_ds2 & resolved & same_ord
                if not can_upd.any():
                    break
                ROW_conf[ur[can_upd], uc[can_upd]] = ROW_conf[nr[can_upd], nc[can_upd]]
                COL_conf[ur[can_upd], uc[can_upd]] = COL_conf[nr[can_upd], nc[can_upd]]

            return ROW_conf, COL_conf

        def compute_gfi_base(fa_arr, row_ch, col_ch, channel):
            """GFI v1: ln(hr / H)  - uses variant-specific fa_arr."""
            valid = ~np.isnan(row_ch)
            R_idx = row_ch[valid].astype(int)
            C_idx = col_ch[valid].astype(int)

            H = np.full((nrows, ncols), np.nan, dtype=np.float32)
            H[valid]       = (dem_arr[valid] - dem_arr[R_idx, C_idx]).astype(np.float32)
            H[channel > 0] = 0.001
            H[H <= 0]      = 0.001

            Ariver = np.full((nrows, ncols), np.nan, dtype=np.float32)
            Ariver[valid]       = fa_arr[R_idx, C_idx].astype(np.float32)
            Ariver[channel > 0] = fa_arr[channel > 0].astype(np.float32)

            Hr = (((Ariver + 1.0) * cell_area) / 1000000.0) ** n_exp

            with np.errstate(divide="ignore", invalid="ignore"):
                GFI_base = np.where(
                    (Hr > 0) & (H > 0),
                    np.log(Hr / H).astype(np.float32),
                    np.nan
                ).astype(np.float32)

            finite  = GFI_base[np.isfinite(GFI_base)]
            n_Hle0  = int(np.sum((H == 0.001) & ~np.isnan(H)))
            messages.addMessage(
                "    MGFI (base) range: [{:.3f}, {:.3f}]  (H<=0 clamped: {})".format(
                    float(finite.min()), float(finite.max()), n_Hle0)
            )
            return GFI_base, Hr.astype(np.float32), H, Ariver

        def _auc_maggiore(fpr_arr, tpr_arr):
            """AUC computed via the Maggiore method (rectangular averaging)."""
            if len(fpr_arr) < 2:
                return 0.0
            x = np.asarray(fpr_arr, dtype=np.float64)
            y = np.asarray(tpr_arr, dtype=np.float64)
            idx = np.argsort(x)
            x = np.concatenate([x[idx], [1.0]])
            y = np.concatenate([y[idx], [1.0]])
            dx = np.diff(x)
            x_diff_full = np.concatenate(([x[0]], dx))
            y_shifted   = np.concatenate(([0.0], y[:-1]))
            auc = (np.sum(y * x_diff_full) + np.sum(y_shifted * x_diff_full)) / 2.0
            return float(auc)

        def _composite_score(auc, perf):
            """Average of AUC, Accuracy, Recall, Precision, F1-score (NaN ignored)."""
            vals = []
            for v in [
                auc,
                (perf or {}).get("Accuracy"),
                (perf or {}).get("Recall (TPR)"),
                (perf or {}).get("Precision"),
                (perf or {}).get("F1-score"),
            ]:
                try:
                    fv = float(v)
                    if not np.isnan(fv):
                        vals.append(fv)
                except (TypeError, ValueError):
                    continue
            return float(np.mean(vals)) if vals else 0.0

        def calibrate_roc(gfi, flood_map, row_ch, col_ch, channel, label):
            """
            ROC calibration via MargArea:
              MargArea[r,c] = the flood_map value at the nearest channel pixel (ROW_ch[r,c]).
              CalibrationArea = all pixels that have a path to a channel (~isnan(ROW_ch))
                                and a finite GFI value.

            Returns: (a_coef, opt_threshold_orig, CalibrationArea, roc_data_dict)
            """
            nan_roc = {
                "fpr": np.array([0.0, 1.0]), "tpr": np.array([0.0, 1.0]),
                "auc": np.nan, "opt_t_norm": 0.0, "opt_t_orig": 0.0,
                "opt_fpr": np.nan, "opt_tpr": np.nan, "opt_dist": np.nan,
                "thresholds_norm": np.array([]), "thresholds_orig": np.array([]),
            }

            # -- Build MargArea: project flood_map onto the nearest channel ------
            valid_idx = ~np.isnan(row_ch)
            R_idx = row_ch[valid_idx].astype(int)
            C_idx = col_ch[valid_idx].astype(int)

            MargArea = np.full((nrows, ncols), np.nan, dtype=np.float32)
            MargArea[valid_idx]   = flood_map[R_idx, C_idx]
            MargArea[channel > 0] = flood_map[channel > 0]

            # CalibrationArea = all pixels with a valid MargArea & finite GFI
            CalibrationArea = np.where(
                ~np.isnan(MargArea) & np.isfinite(gfi), 1.0, 0.0
            ).astype(np.float32)

            n_calib  = int((CalibrationArea == 1).sum())
            n_flood  = int(((MargArea == 1) & (CalibrationArea == 1)).sum())
            n_dry    = int(((MargArea == 0) & (CalibrationArea == 1)).sum())
            pct      = n_calib / float(nrows * ncols) * 100.0
            messages.addMessage(
                "  [CAL {}] MargArea - CA: {} px ({:.4f}%) | flooded: {} | dry: {}".format(
                    label, n_calib, pct, n_flood, n_dry)
            )

            if n_calib < 10:
                messages.addMessage(
                    "  [ROC {}] Insufficient CA pixels - calibration skipped.".format(label)
                )
                return 1.0, 0.0, CalibrationArea, nan_roc

            # -- ROC sweep within the CalibrationArea ----------------------------------
            ca_mask   = (CalibrationArea == 1)
            gfi_eval  = gfi[ca_mask].astype(np.float64)
            fh_eval   = np.where(flood_map[ca_mask] > 0, 1, 0).astype(np.int8)

            P_px    = int((fh_eval == 1).sum())
            N_px    = int((fh_eval == 0).sum())
            n_valid = P_px + N_px
            messages.addMessage(
                "  [ROC {}] Eval pixels: {}  (flooded={} dry={})".format(
                    label, n_valid, P_px, N_px)
            )

            if n_valid < 10 or P_px == 0 or N_px == 0:
                messages.addMessage(
                    "  [ROC {}] Insufficient pixels (P={}, N={}) - calibration skipped.".format(
                        label, P_px, N_px)
                )
                return 1.0, 0.0, CalibrationArea, nan_roc

            if float(P_px) / float(N_px) > 20.0 or float(P_px) / float(N_px) < 0.05:
                messages.addMessage(
                    "  [ROC {}] WARNING: severe class imbalance (flooded/dry={:.1f}).".format(
                        label, float(P_px) / float(N_px))
                )

            # Normalise GFI -> [-1, 1]
            gfi_min = float(np.nanmin(gfi_eval))
            gfi_max = float(np.nanmax(gfi_eval))
            if gfi_max == gfi_min:
                messages.addMessage("  [ROC {}] GFI has no variation.".format(label))
                return 1.0, 0.0, CalibrationArea, nan_roc

            gfi_norm = 2.0 * ((gfi_eval - gfi_min) / (gfi_max - gfi_min) - 0.5)

            # Sweep threshold from -1 to +1 (arange)
            thresholds_norm = np.arange(-1.0, 1.0 + roc_steps, roc_steps)
            fpr_arr = np.empty(len(thresholds_norm), dtype=np.float64)
            tpr_arr = np.empty(len(thresholds_norm), dtype=np.float64)

            best_dist  = np.inf
            opt_t_norm = thresholds_norm[0]
            opt_t_orig = 0.0
            opt_fpr    = 1.0
            opt_tpr    = 0.0

            truth_vals = fh_eval
            for i, t in enumerate(thresholds_norm):
                pred = (gfi_norm >= t).astype(np.int8)
                TP = int(((pred == 1) & (truth_vals == 1)).sum())
                TN = int(((pred == 0) & (truth_vals == 0)).sum())
                FP = int(((pred == 1) & (truth_vals == 0)).sum())
                FN = int(((pred == 0) & (truth_vals == 1)).sum())

                fpr = FP / float(FP + TN) if (FP + TN) > 0 else 0.0
                fnr = FN / float(FN + TP) if (FN + TP) > 0 else 0.0
                tpr = 1.0 - fnr

                fpr_arr[i] = fpr
                tpr_arr[i] = tpr

                dist = fpr + fnr   # minimize FPR + FNR
                if dist < best_dist:
                    best_dist  = dist
                    opt_t_norm = t
                    opt_fpr    = fpr
                    opt_tpr    = tpr
                    opt_t_orig = ((t + 1.0) / 2.0) * (gfi_max - gfi_min) + gfi_min

            a       = float(np.exp(-opt_t_orig))
            auc_val = _auc_maggiore(fpr_arr, tpr_arr)
            thresholds_orig = ((thresholds_norm + 1.0) / 2.0) * (gfi_max - gfi_min) + gfi_min

            roc_data = {
                "fpr": fpr_arr, "tpr": tpr_arr,
                "thresholds_norm": thresholds_norm,
                "thresholds_orig": thresholds_orig,
                "auc": auc_val,
                "opt_t_norm": opt_t_norm, "opt_t_orig": opt_t_orig,
                "opt_fpr": opt_fpr, "opt_tpr": opt_tpr, "opt_dist": best_dist,
            }

            messages.addMessage(
                "  [ROC {}] AUC={:.4f}  t_opt={:.4f}  a={:.6f}  "
                "FPR={:.4f}  TPR={:.4f}".format(
                    label, auc_val, opt_t_orig, a, opt_fpr, opt_tpr)
            )
            return a, opt_t_orig, CalibrationArea, roc_data

        def compute_gfi_backwater(fa_arr, channel, strahler, row_ch, col_ch,
                                   row_conf, col_conf, a_v1):
            """GFI v2: hierarchical Strahler backwater - uses variant-specific fa_arr."""
            Ariver_net    = fa_arr.copy().astype(np.float32)
            DEM_river_net = dem_arr.copy().astype(np.float32)

            hr_net = (((Ariver_net + 1.0) * cell_area) / 1000000.0) ** n_exp
            WD_net = np.maximum(0.0, hr_net * a_v1 - 0.001)

            curr_row_conf = row_conf.copy()
            curr_col_conf = col_conf.copy()

            ch_pixels     = list(zip(*np.where(channel > 0)))
            total_updated = 0

            for k in range(v2_max_iter):
                updated = 0
                for r, c in ch_pixels:
                    nr_f = curr_row_conf[r, c]
                    nc_f = curr_col_conf[r, c]
                    if np.isnan(nr_f):
                        continue

                    nr, nc   = int(nr_f), int(nc_f)
                    A_conf   = float(fa_arr[nr, nc])
                    hr_conf  = (((A_conf + 1.0) * cell_area) / 1000000.0) ** n_exp

                    H_to_conf = max(0.001, float(dem_arr[r, c]) - float(dem_arr[nr, nc]))

                    WD_potential = max(0.0, hr_conf * a_v1 - H_to_conf)

                    if WD_potential > WD_net[r, c]:
                        Ariver_net[r, c]    = A_conf
                        DEM_river_net[r, c] = float(dem_arr[nr, nc])
                        WD_net[r, c]        = WD_potential
                        curr_row_conf[r, c] = row_conf[nr, nc]
                        curr_col_conf[r, c] = col_conf[nr, nc]
                        updated += 1

                total_updated += updated
                messages.addMessage(
                    "    Backwater iteration {}/{}: {} pixels updated".format(
                        k + 1, v2_max_iter, updated)
                )
                if updated == 0:
                    break

            valid = ~np.isnan(row_ch)
            R_idx = row_ch[valid].astype(int)
            C_idx = col_ch[valid].astype(int)

            Ariver_bw = np.full((nrows, ncols), np.nan, dtype=np.float32)
            H_bw      = np.full((nrows, ncols), np.nan, dtype=np.float32)

            Ariver_bw[valid]       = Ariver_net[R_idx, C_idx]
            Ariver_bw[channel > 0] = Ariver_net[channel > 0]

            H_bw[valid]       = dem_arr[valid] - DEM_river_net[R_idx, C_idx]
            H_bw[channel > 0] = dem_arr[channel > 0] - DEM_river_net[channel > 0]
            H_bw[H_bw <= 0]   = 0.001
            Hr_bw = (((Ariver_bw + 1.0) * cell_area) / 1000000.0) ** n_exp

            with np.errstate(divide="ignore", invalid="ignore"):
                GFI_bw = np.where(
                    (Hr_bw > 0) & (H_bw > 0),
                    np.log(Hr_bw / H_bw).astype(np.float32),
                    np.nan
                ).astype(np.float32)

            finite2 = GFI_bw[np.isfinite(GFI_bw)]
            messages.addMessage(
                "    MGFI (backwater) range: [{:.3f}, {:.3f}]  total px updated={}".format(
                    float(finite2.min()), float(finite2.max()), total_updated)
            )
            return GFI_bw, Hr_bw, H_bw, Ariver_bw

        def compute_performance(flood_pred, flood_map, marg_area, label, roc_data=None):
            valid   = (marg_area == 1) & ~np.isnan(flood_map) & ~np.isnan(flood_pred)
            n_valid = int(valid.sum())
            if n_valid == 0:
                messages.addMessage(
                    "  [PERF {}] No valid evaluation pixels.".format(label)
                )
                return {
                    "version": label,
                    "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                    "Accuracy": np.nan, "Precision": np.nan,
                    "Recall (TPR)": np.nan, "FNR": np.nan,
                    "TNR (Spec.)": np.nan, "FPR": np.nan,
                    "F1-score": np.nan, "CSI": np.nan,
                    "Kappa": np.nan, "Bias": np.nan,
                    "Sum (FPR+FNR)": np.nan, "AUC": np.nan,
                }
            pred  = flood_pred[valid].astype(int)
            truth = flood_map[valid].astype(int)
            n_pos = int((truth == 1).sum())
            n_neg = n_valid - n_pos
            messages.addMessage(
                "  [PERF {}] Eval domain: {} pixels  (flooded={} dry={})".format(
                    label, n_valid, n_pos, n_neg)
            )

            TP = int(((pred == 1) & (truth == 1)).sum())
            TN = int(((pred == 0) & (truth == 0)).sum())
            FP = int(((pred == 1) & (truth == 0)).sum())
            FN = int(((pred == 0) & (truth == 1)).sum())
            total = TP + TN + FP + FN

            def _safe_div(num, den):
                return round(num / den, 4) if den > 0 else np.nan

            Accuracy  = _safe_div(TP + TN, total)
            Precision = _safe_div(TP, TP + FP)          # same as PPV
            TPR       = _safe_div(TP, TP + FN)          # Recall
            TNR       = _safe_div(TN, TN + FP)          # Specificity
            FPR       = _safe_div(FP, FP + TN)
            FNR       = _safe_div(FN, FN + TP)
            F1        = _safe_div(2 * TP, 2 * TP + FP + FN)
            CSI       = _safe_div(TP, TP + FP + FN)
            Bias      = _safe_div(TP + FP, TP + FN)
            Sum_dist  = round(FPR + FNR, 4) if not (np.isnan(FPR) or np.isnan(FNR)) else np.nan

            p_obs = (TP + TN) / total if total > 0 else np.nan
            p_exp = (((TP + FP) * (TP + FN)) + ((TN + FN) * (TN + FP))) / total ** 2 \
                    if total > 0 else np.nan
            Kappa = round((p_obs - p_exp) / (1.0 - p_exp), 4) \
                    if (p_exp is not None and not np.isnan(p_exp) and p_exp < 1.0) else np.nan

            AUC = round(float(roc_data["auc"]), 4) \
                  if (roc_data is not None and not np.isnan(roc_data["auc"])) else np.nan

            messages.addMessage(
                "  [MGFI {}] AUC={}  Accuracy={}  Precision={}  Recall(TPR)={}  "
                "F1={}  CSI={}  Kappa={}  TNR={}  FPR={}  FNR={}  Sum={}".format(
                    label, AUC, Accuracy, Precision, TPR,
                    F1, CSI, Kappa, TNR, FPR, FNR, Sum_dist)
            )
            # Column order matches the standard compute_metrics layout
            return {
                "version":       label,
                "TP":            TP,
                "TN":            TN,
                "FP":            FP,
                "FN":            FN,
                "Accuracy":      Accuracy,
                "Precision":     Precision,
                "Recall (TPR)":  TPR,
                "FNR":           FNR,
                "TNR (Spec.)":   TNR,
                "FPR":           FPR,
                "F1-score":      F1,
                "CSI":           CSI,
                "Kappa":         Kappa,
                "Bias":          Bias,
                "Sum (FPR+FNR)": Sum_dist,
                "AUC":           AUC,
            }

        def calc_water_depth(hr_arr, H_arr, a_coef, flood_bin):
            H_safe = H_arr.copy().astype(np.float64)
            H_safe[H_safe <= 0] = np.nan
            with np.errstate(invalid="ignore"):
                WD = np.maximum(0.0, hr_arr.astype(np.float64) * a_coef - H_safe)
            WD = WD.astype(np.float32)
            WD[flood_bin != 1]    = np.nan
            WD[np.isnan(dem_arr)] = np.nan
            return WD

        def classify_flood_hazard(wd_arr, variant, version):
            cls_arr = np.zeros(wd_arr.shape, dtype=np.int16)
            valid   = ~np.isnan(wd_arr)
            cls_arr[valid] = (np.searchsorted([0.75, 1.5], wd_arr[valid], side="right") + 1).astype(np.int16)
            if not valid.any():
                messages.addMessage("  Hazard raster: no valid pixels.")
                return None
            hzd_path = make_path("_Hazard_class", variant, version)
            save_raster_int(cls_arr, hzd_path, nodata=0)
            n_cls = {k: int((cls_arr == k).sum()) for k in [1, 2, 3]}
            messages.addMessage(
                "  Hazard class v{}: Low={} Medium={} High={} -> {}".format(
                    version, n_cls[1], n_cls[2], n_cls[3], hzd_path)
            )
            return hzd_path

        def get_flood_prone_mask(gfi_arr, threshold):
            # v10.1: strict '>' threshold and 8-connectivity structure
            fb      = np.where(np.isfinite(gfi_arr), (gfi_arr > threshold).astype(np.float32), np.nan)
            fb_bin  = np.where(np.isfinite(fb), fb, 0).astype(np.int32)
            struct8 = np.ones((3, 3), dtype=int)
            labelled, n_lab = ndimage.label(fb_bin, structure=struct8)
            sizes   = ndimage.sum(fb_bin, labelled, range(n_lab + 1))
            cleaned = np.zeros((nrows, ncols), dtype=np.float32)
            for lbl in np.where(sizes >= 8)[0]:
                cleaned[labelled == lbl] = 1.0
            cleaned[np.isnan(dem_arr)] = 0.0
            return cleaned

        def point_accuracy(gfi_arr, threshold, pts_shp):
            if not pts_shp:
                return None, None
            try:
                pred_arr = np.where(
                    np.isfinite(gfi_arr), (gfi_arr >= threshold).astype(np.float32), 0.0
                )
                pred_arr[np.isnan(dem_arr)] = 0.0
                tmp_path = os.path.join(out_folder, "_tmp_pt_check.tif")
                save_raster_float(pred_arr, tmp_path)

                n_hit = n_total = 0
                with arcpy.da.SearchCursor(pts_shp, ["SHAPE@XY"]) as cur:
                    for row in cur:
                        xp, yp = row[0]
                        try:
                            cell_str = arcpy.management.GetCellValue(
                                tmp_path, "{} {}".format(xp, yp)
                            ).getOutput(0)
                            n_total += 1
                            if cell_str not in ("NoData", "", None):
                                val = float(cell_str)
                                if val > 0:
                                    n_hit += 1
                        except Exception:
                            pass

                if arcpy.Exists(tmp_path):
                    arcpy.Delete_management(tmp_path)
                if n_total > 0:
                    return float(n_hit) / n_total, n_hit
            except Exception as ex:
                messages.addMessage("Point accuracy check failed: {}".format(str(ex)))
            return None, None

        # -- Load SFM ----------------------------------------------------------
        sfm_arr = None
        if do_calibrate and sfm_raster:
            sfm_r   = arcpy.Raster(sfm_raster)
            sfm_ndv = sfm_r.noDataValue if sfm_r.noDataValue is not None else -9999
            sfm_raw = arcpy.RasterToNumPyArray(
                sfm_raster, nodata_to_value=sfm_ndv
            ).astype(np.float32)
            sfm_raw[sfm_raw == sfm_ndv] = np.nan
            sfm_arr = np.where(sfm_raw > 0, 1.0, 0.0).astype(np.float32)
            sfm_arr[np.isnan(sfm_raw)] = np.nan

        _fa_for_val = fa_array_map.get("CONVENTIONAL", fa_array_map[variant_list[0]])
        validate_inputs(dem_arr, _fa_for_val, flow_dir_arr, sfm_arr)

        # ======================================================================
        # STEP 4-7 - Main loop per variant
        # ======================================================================
        messages.addMessage("=" * 60)
        channel_mode_label = "Channel_ASk" if use_ask else "Channel_GFI20"
        messages.addMessage(
            "STEP 4: Channel delineation & MGFI  [mode: {}]".format(channel_mode_label)
        )

        results_summary = {}

        for variant in variant_list:
            t_start = time.time()
            messages.addMessage("")
            messages.addMessage(">>> Variant: {} <<<".format(LABEL[variant]))

            fa_arr = fa_array_map[variant]

            # Choose candidate list for auto-iteration
            # Both Horn-based modes share the same candidate scale
            candidates = HORN_CANDIDATES if do_auto_ci else [ci_threshold]

            if do_auto_ci and sfm_arr is not None:
                messages.addMessage(
                    "  [AUTO CI] Testing {} thresholds: {}".format(
                        len(candidates), candidates)
                )

            best_ci      = candidates[0]
            best_metrics = None
            best         = {}
            seed_log     = []
            # Cache the channel index arrays for this variant (same FA, recompute only once)
            _idx_cache = {}

            for ci_val in candidates:
                messages.addMessage(
                    "  Building channel network ({} threshold={})...".format(
                        channel_mode_label, int(ci_val))
                )
                channel_arr, ask_arr, gfi20_arr = build_channel_network(fa_arr, ci_val)

                if "ask" not in _idx_cache:
                    _idx_cache["ask"]   = ask_arr
                    _idx_cache["gfi20"] = gfi20_arr

                messages.addMessage("  Computing Strahler order...")
                strahler_arr = compute_strahler(channel_arr)

                messages.addMessage("  Mapping hillslope -> nearest channel pixel...")
                row_ch_arr, col_ch_arr = map_hillslope_to_channel(channel_arr)

                messages.addMessage("  Mapping channel -> next confluence...")
                row_conf_arr, col_conf_arr = map_channel_to_confluence(channel_arr, strahler_arr)

                messages.addMessage("  Computing MGFI (base)...")
                gfi1, hr1, h1, ar1 = compute_gfi_base(fa_arr, row_ch_arr, col_ch_arr, channel_arr)

                if do_auto_ci and sfm_arr is not None:
                    # --- Step A: calibrate MGFI (base/v1) just to obtain a_init ---
                    # for the backwater run. This threshold/a1 is NOT used for
                    # scoring the candidate anymore (see note below).
                    messages.addMessage("  Calibrating MGFI (base) to get a_init...")
                    a1, t1, marg1, roc1 = calibrate_roc(
                        gfi1, sfm_arr, row_ch_arr, col_ch_arr, channel_arr, "base"
                    )

                    # --- Step B: run backwater (v2/GFI20) for THIS candidate ---
                    # NOTE (patched): CI-threshold auto-iterate now scores each
                    # candidate on the POST-BACKWATER result (GFI20), not on the
                    # pre-backwater GFI v1 as before. This mirrors the Colab
                    # sweep (_eval_th_channel_mod), which evaluates AUC/Accuracy
                    # on gfi_s3 (post-backwater) for every TH_CHANNEL_MOD
                    # candidate. Rationale: the reported/final product is the
                    # backwater-adjusted map, so the channel network that is
                    # selected should be the one that performs best on that
                    # final product, not on an intermediate stage.
                    messages.addMessage("  Computing MGFI (backwater) for CI comparison...")
                    GFI20_cand, hr2_cand, h2_cand, ar2_cand = compute_gfi_backwater(
                        fa_arr, channel_arr, strahler_arr,
                        row_ch_arr, col_ch_arr, row_conf_arr, col_conf_arr,
                        a1
                    )

                    messages.addMessage("  Calibrating MGFI (backwater) for CI comparison...")
                    a2_cand, t2_cand, marg2_cand, roc2_cand = calibrate_roc(
                        GFI20_cand, sfm_arr, row_ch_arr, col_ch_arr, channel_arr, "backwater"
                    )
                    fp2_cand = get_flood_prone_mask(GFI20_cand, t2_cand)
                    met = compute_performance(fp2_cand, sfm_arr, marg2_cand, "backwater", roc2_cand)
                    cur_score = _composite_score(met["AUC"], met)
                    seed_log.append({
                        "ci": int(ci_val), "auc": met["AUC"] or 0.0,
                        "accuracy": met["Accuracy"] or 0.0,
                        "recall": met["Recall (TPR)"] or 0.0,
                        "precision": met["Precision"] or 0.0,
                        "f1": met["F1-score"] or 0.0,
                        "csi": met["CSI"] or 0.0, "kappa": met["Kappa"] or 0.0,
                        "score": cur_score,
                    })
                    if best_metrics is None or cur_score > best_metrics.get("_score", 0.0):
                        best_metrics = dict(met)
                        best_metrics["_score"] = cur_score
                        best_ci = ci_val
                        best = {
                            "channel": channel_arr.copy(), "strahler": strahler_arr.copy(),
                            "row_ch": row_ch_arr.copy(), "col_ch": col_ch_arr.copy(),
                            "row_conf": row_conf_arr.copy(), "col_conf": col_conf_arr.copy(),
                            "gfi1": gfi1.copy(), "hr1": hr1.copy(), "h1": h1.copy(),
                            "a1": a1, "t1": t1, "marg": marg1.copy(), "roc1": dict(roc1),
                            # cached post-backwater result for the winning
                            # candidate, so it does not need to be recomputed
                            # after the loop.
                            "GFI20": GFI20_cand.copy(), "hr2": hr2_cand.copy(),
                            "h2": h2_cand.copy(),
                            "a2": a2_cand, "t2": t2_cand,
                            "marg2": marg2_cand.copy(), "roc2": dict(roc2_cand),
                        }
                else:
                    best = {
                        "channel": channel_arr, "strahler": strahler_arr,
                        "row_ch": row_ch_arr, "col_ch": col_ch_arr,
                        "row_conf": row_conf_arr, "col_conf": col_conf_arr,
                        "gfi1": gfi1, "hr1": hr1, "h1": h1,
                        "a1": 1.0, "t1": manual_thr_val,
                        "marg": None, "roc1": None,
                    }
                    best_ci = ci_val

            # CI comparison table (scored on POST-BACKWATER / GFI20 result)
            if do_auto_ci and len(seed_log) > 1:
                messages.addMessage("  " + "-" * 100)
                messages.addMessage(
                    "  CI comparison — metrics computed AFTER backwater (GFI20), "
                    "matching the final reported product."
                )
                messages.addMessage(
                    "  {:>10}  {:>8}  {:>8}  {:>8}  {:>9}  {:>8}  {:>8}  {:>8}  {:>8}".format(
                        "Threshold", "AUC", "Accuracy", "Recall", "Precision",
                        "F1", "CSI", "Kappa", "Score")
                )
                for entry in sorted(seed_log, key=lambda x: x["score"], reverse=True):
                    marker = " << BEST" if entry["ci"] == int(best_ci) else ""
                    messages.addMessage(
                        "  {:>10}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {:>9.4f}  {:>8.4f}  "
                        "{:>8.4f}  {:>8.4f}  {:>8.4f}{}".format(
                            entry["ci"], entry["auc"], entry["accuracy"], entry["recall"],
                            entry["precision"], entry["f1"], entry["csi"], entry["kappa"],
                            entry["score"], marker)
                    )
                messages.addMessage("  " + "-" * 100)

            # Final GFI v1 calibration
            if do_calibrate and sfm_arr is not None:
                if not do_auto_ci:
                    messages.addMessage("  Calibrating MGFI (base, final)...")
                    best["a1"], best["t1"], best["marg"], best["roc1"] = calibrate_roc(
                        best["gfi1"], sfm_arr, best["row_ch"], best["col_ch"],
                        best["channel"], "1.0"
                    )
            elif use_manual_thr:
                best["t1"] = manual_thr_val
                best["a1"] = float(np.exp(-manual_thr_val))

            # GFI v2: backwater
            if do_auto_ci and sfm_arr is not None and "GFI20" in best:
                # Already computed + calibrated for the winning CI candidate
                # inside the auto-iterate loop above -> reuse it, don't
                # recompute (also keeps the reported metrics identical to
                # what was used to pick this threshold).
                messages.addMessage("  Using cached MGFI (backwater) from CI auto-iteration...")
                GFI20 = best["GFI20"]; hr2 = best["hr2"]; h2 = best["h2"]
                a2    = best["a2"];    t2  = best["t2"]
                marg2 = best["marg2"]; roc2 = best["roc2"]
            else:
                messages.addMessage("  Computing MGFI (backwater effect)...")
                GFI20, hr2, h2, ar2 = compute_gfi_backwater(
                    fa_arr, best["channel"], best["strahler"],
                    best["row_ch"], best["col_ch"],
                    best["row_conf"], best["col_conf"],
                    best["a1"]
                )

                a2    = best["a1"]
                t2    = best["t1"]
                marg2 = best["marg"]
                roc2  = best["roc1"]

                if do_calibrate and sfm_arr is not None:
                    messages.addMessage("  Calibrating MGFI (backwater)...")
                    a2, t2, marg2, roc2 = calibrate_roc(
                        GFI20, sfm_arr, best["row_ch"], best["col_ch"], best["channel"], "backwater"
                    )

            # Binary flood masks
            flood_v1 = get_flood_prone_mask(best["gfi1"], best["t1"])
            flood_v2 = get_flood_prone_mask(GFI20, t2)

            # Performance
            perf_v1 = perf_v2 = None
            if do_calibrate and sfm_arr is not None and best["marg"] is not None:
                perf_v1 = compute_performance(flood_v1, sfm_arr, best["marg"],  "base", best["roc1"])
                _marg_for_v2 = marg2 if marg2 is not None else best["marg"]
                perf_v2 = compute_performance(flood_v2, sfm_arr, _marg_for_v2, "backwater", roc2)

            # Point accuracy
            pt_acc_v1 = pt_hit_v1 = None
            pt_acc_v2 = pt_hit_v2 = None
            if do_calibrate and flood_points:
                pt_acc_v1, pt_hit_v1 = point_accuracy(best["gfi1"], best["t1"], flood_points)
                pt_acc_v2, pt_hit_v2 = point_accuracy(GFI20, t2, flood_points)
                for ver, pa, ph in [("v1", pt_acc_v1, pt_hit_v1), ("v2", pt_acc_v2, pt_hit_v2)]:
                    if pa is not None:
                        n_total = int(ph / pa) if pa > 0 else 0
                        messages.addMessage(
                            "  Point accuracy MGFI {}: {}/{} points ({:.2f}%)".format(
                                ver, ph, n_total, pa * 100.0)
                        )

            # -- Save output rasters ------------------------------------------
            messages.addMessage("  Saving output rasters...")

            channel_path  = make_path("_Channel",     variant)
            h_v1_path     = make_path("_H",           variant, "1")
            h_v2_path     = make_path("_H",           variant, "2")
            hr_v1_path    = make_path("_Hr",          variant, "1")
            hr_v2_path    = make_path("_Hr",          variant, "2")
            gfi1_path     = make_path("_GFI",         variant, "1")
            GFI20_path     = make_path("_GFI",         variant, "2")
            flood_v1_path = make_path("_Floodhazard", variant, "1")
            flood_v2_path = make_path("_Floodhazard", variant, "2")

            save_raster_int(best["channel"], channel_path, nodata=-9999)
            save_raster_float(best["h1"],   h_v1_path)
            save_raster_float(h2,           h_v2_path)
            save_raster_float(best["hr1"],  hr_v1_path)
            save_raster_float(hr2,          hr_v2_path)
            save_raster_float(best["gfi1"], gfi1_path)
            save_raster_float(GFI20,         GFI20_path)
            save_raster_float(flood_v1,     flood_v1_path)
            save_raster_float(flood_v2,     flood_v2_path)

            strahler_path = None
            if do_strahler:
                strahler_path = make_path("_Strahler", variant)
                save_raster_int(best["strahler"], strahler_path, nodata=0)

            marg_path = None
            if do_calibrate and best["marg"] is not None:
                marg_path = make_path("_MargArea", variant)
                save_raster_float(best["marg"], marg_path)

            # Export the selected channel index raster (diagnostic)
            channel_idx_path = None
            if do_export_channel_idx and "ask" in _idx_cache:
                idx_key   = "ask" if use_ask else "gfi20"
                channel_idx_path = make_path("_{}".format(channel_mode_label), variant)
                save_raster_float(_idx_cache[idx_key], channel_idx_path)
                messages.addMessage("  {} index -> {}".format(channel_mode_label, channel_idx_path))

            if save_intermediate:
                save_raster_float(best["row_ch"],   make_path("_Row_ch",   variant))
                save_raster_float(best["col_ch"],   make_path("_Col_ch",   variant))
                save_raster_float(best["row_conf"], make_path("_Row_conf", variant))
                save_raster_float(best["col_conf"], make_path("_Col_conf", variant))

            wd_v1_path = wd_v2_path = hzd_v1_path = hzd_v2_path = None
            if do_water_depth:
                messages.addMessage("  Computing Water Depth (base)...")
                WD_v1      = calc_water_depth(best["hr1"], best["h1"], best["a1"], flood_v1)
                wd_v1_path = make_path("_WaterDepth", variant, "1")
                save_raster_float(WD_v1, wd_v1_path)
                hzd_v1_path = classify_flood_hazard(WD_v1, variant, "1")

                messages.addMessage("  Computing Water Depth (backwater)...")
                WD_v2      = calc_water_depth(hr2, h2, a2, flood_v2)
                wd_v2_path = make_path("_WaterDepth", variant, "2")
                save_raster_float(WD_v2, wd_v2_path)
                hzd_v2_path = classify_flood_hazard(WD_v2, variant, "2")

            # -- Performance report TXT ----------------------------------------
            if do_calibrate and (perf_v1 or perf_v2):
                tag_sfx   = "_{}".format(TAG[variant]) if not single_variant else ""
                perf_path = os.path.join(
                    out_folder, "{}{}_performance.txt".format(out_prefix, tag_sfx)
                )
                try:
                    with io.open(perf_path, "w", encoding="utf-8") as pf:
                        pf.write("MGFI Toolbox v10 Performance Report\n")
                        pf.write("Channel_mode\t{}\n".format(channel_mode_label))
                        pf.write("CI_threshold\t{}\n".format(int(best_ci)))
                        pf.write("exponent_n\t{}\n".format(n_exp))
                        pf.write("v2_max_iter\t{}\n".format(v2_max_iter))
                        pf.write("roc_step_size\t{}\n".format(roc_steps))
                        pf.write("\n--- MGFI Base ---\n")
                        pf.write("threshold\t{:.10f}\n".format(best["t1"]))
                        pf.write("a_coefficient\t{:.10f}\n".format(best["a1"]))
                        if best["roc1"]:
                            pf.write("AUC_v1\t{:.10f}\n".format(best["roc1"]["auc"]))
                            pf.write("opt_FPR_v1\t{:.10f}\n".format(best["roc1"]["opt_fpr"]))
                            pf.write("opt_TPR_v1\t{:.10f}\n".format(best["roc1"]["opt_tpr"]))
                        if perf_v1:
                            for k, v in perf_v1.items():
                                if k != "version":
                                    pf.write("{}_v1\t{}\n".format(k, v))
                        if pt_acc_v1 is not None:
                            pf.write("PointAccuracy_v1\t{:.6f}\n".format(pt_acc_v1))
                        pf.write("\n--- MGFI Backwater Effect ---\n")
                        pf.write("threshold_v2\t{:.10f}\n".format(t2))
                        pf.write("a_coefficient_v2\t{:.10f}\n".format(a2))
                        if roc2:
                            pf.write("AUC_v2\t{:.10f}\n".format(roc2["auc"]))
                            pf.write("opt_FPR_v2\t{:.10f}\n".format(roc2["opt_fpr"]))
                            pf.write("opt_TPR_v2\t{:.10f}\n".format(roc2["opt_tpr"]))
                        if perf_v2:
                            for k, v in perf_v2.items():
                                if k != "version":
                                    pf.write("{}_v2\t{}\n".format(k, v))
                        if pt_acc_v2 is not None:
                            pf.write("PointAccuracy_v2\t{:.6f}\n".format(pt_acc_v2))
                        if do_auto_ci:
                            pf.write("\n--- Auto-Iteration ---\n")
                            pf.write(
                                "SelectionCriterion\tScore = mean(AUC, Accuracy, "
                                "Recall, Precision, F1-score)\n"
                            )
                            pf.write("BestThreshold\t{}\n".format(int(best_ci)))
                            for entry in seed_log:
                                pf.write(
                                    "thr_{}\tAUC={:.6f}\tAccuracy={:.6f}\tRecall={:.6f}\t"
                                    "Precision={:.6f}\tF1={:.6f}\tCSI={:.6f}\tKappa={:.6f}\t"
                                    "Score={:.6f}\n".format(
                                        entry["ci"], entry["auc"], entry["accuracy"],
                                        entry["recall"], entry["precision"], entry["f1"],
                                        entry["csi"], entry["kappa"], entry["score"])
                                )
                    messages.addMessage("  Performance report -> {}".format(perf_path))
                except Exception as ex:
                    messages.addMessage("Could not save performance.txt: {}".format(str(ex)))

            # -- Performance report CSV ----------------------------------------
            if do_export_csv and do_calibrate and (perf_v1 or perf_v2):
                import csv
                from datetime import datetime
                tag_sfx  = "_{}".format(TAG[variant]) if not single_variant else ""
                csv_path = os.path.join(
                    out_folder, "{}{}_performance.csv".format(out_prefix, tag_sfx)
                )
                # Explicit columns - standard compute_metrics column order
                CSV_FIELDS = [
                    "timestamp", "variant", "version",
                    "channel_mode", "ci_threshold", "n_exp", "v2_iter", "roc_step",
                    "threshold", "a_coef",
                    "AUC", "opt_FPR", "opt_TPR",
                    "TP", "TN", "FP", "FN",
                    "Accuracy", "Precision", "Recall (TPR)", "FNR",
                    "TNR (Spec.)", "FPR", "F1-score", "CSI",
                    "Kappa", "Bias", "Sum (FPR+FNR)", "Score",
                ]
                try:
                    csv_rows = []
                    for ver, roc_c, perf, thr, a_c in [
                        ("base",      best["roc1"], perf_v1, best["t1"], best["a1"]),
                        ("backwater", roc2,         perf_v2, t2,         a2),
                    ]:
                        if perf is None:
                            continue
                        _row = {
                            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "variant":      LABEL[variant],
                            "version":      ver,
                            "channel_mode": channel_mode_label,
                            "ci_threshold": int(best_ci),
                            "n_exp":        n_exp,
                            "v2_iter":      v2_max_iter,
                            "roc_step":     roc_steps,
                            "threshold":    round(thr, 6),
                            "a_coef":       round(a_c, 6),
                            "AUC":          round(roc_c["auc"], 4)     if roc_c else None,
                            "opt_FPR":      round(roc_c["opt_fpr"], 4) if roc_c else None,
                            "opt_TPR":      round(roc_c["opt_tpr"], 4) if roc_c else None,
                            "TP":           perf.get("TP"),
                            "TN":           perf.get("TN"),
                            "FP":           perf.get("FP"),
                            "FN":           perf.get("FN"),
                            "Accuracy":     perf.get("Accuracy"),
                            "Precision":    perf.get("Precision"),
                            "Recall (TPR)": perf.get("Recall (TPR)"),
                            "FNR":          perf.get("FNR"),
                            "TNR (Spec.)":  perf.get("TNR (Spec.)"),
                            "FPR":          perf.get("FPR"),
                            "F1-score":     perf.get("F1-score"),
                            "CSI":          perf.get("CSI"),
                            "Kappa":        perf.get("Kappa"),
                            "Bias":         perf.get("Bias"),
                            "Sum (FPR+FNR)":perf.get("Sum (FPR+FNR)"),
                            "Score":        round(
                                _composite_score(roc_c["auc"] if roc_c else None, perf), 4
                            ),
                        }
                        csv_rows.append(_row)
                    if csv_rows:
                        with open(csv_path, "wb") as f:
                            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                            writer.writeheader()
                            writer.writerows(csv_rows)
                        messages.addMessage("  Performance CSV -> {}".format(csv_path))
                except Exception as ex:
                    messages.addMessage("Could not save CSV: {}".format(repr(ex)))

            # -- ROC curve data export (NPZ + CSV) ------------------------------
            # Stored as a cached .npz/.csv pair so the ROC curve can be
            # re-plotted later without rerunning the full calculation.
            if do_calibrate and (best["roc1"] or roc2):
                import csv as _csv
                tag_sfx = "_{}".format(TAG[variant]) if not single_variant else ""

                for ver, roc_c in [("base", best["roc1"]), ("backwater", roc2)]:
                    if not roc_c or roc_c.get("fpr") is None or len(roc_c["fpr"]) == 0:
                        continue

                    # -- NPZ (same format as {variant}_roc.npz) --------
                    npz_path = os.path.join(
                        out_folder, "{}{}_roc_{}.npz".format(out_prefix, tag_sfx, ver)
                    )
                    try:
                        np.savez(
                            npz_path,
                            fpr=roc_c["fpr"], tpr=roc_c["tpr"],
                            thresholds_norm=roc_c["thresholds_norm"],
                            thresholds_orig=roc_c["thresholds_orig"],
                            auc=roc_c["auc"],
                            opt_t_norm=roc_c["opt_t_norm"], opt_t_orig=roc_c["opt_t_orig"],
                            opt_fpr=roc_c["opt_fpr"], opt_tpr=roc_c["opt_tpr"],
                        )
                        messages.addMessage("  ROC curve NPZ ({}) -> {}".format(ver, npz_path))
                    except Exception as ex:
                        messages.addMessage("Could not save ROC NPZ ({}): {}".format(ver, repr(ex)))

                    # -- CSV (one row per threshold, for opening in Excel) ----
                    csv_roc_path = os.path.join(
                        out_folder, "{}{}_roc_{}.csv".format(out_prefix, tag_sfx, ver)
                    )
                    try:
                        with open(csv_roc_path, "wb") as f:
                            w = _csv.writer(f)
                            w.writerow(["threshold_norm", "threshold_orig", "fpr", "tpr"])
                            for tn, to, fp, tp in zip(
                                roc_c["thresholds_norm"], roc_c["thresholds_orig"],
                                roc_c["fpr"], roc_c["tpr"]
                            ):
                                w.writerow([tn, to, fp, tp])
                        messages.addMessage("  ROC curve CSV ({}) -> {}".format(ver, csv_roc_path))
                    except Exception as ex:
                        messages.addMessage("Could not save ROC CSV ({}): {}".format(ver, repr(ex)))

            elapsed = time.time() - t_start
            messages.addMessage(
                "GFI [{}] completed in {:.2f}s".format(LABEL[variant], elapsed)
            )

            results_summary[variant] = {
                "best_ci":   int(best_ci),
                "a_v1":      best["a1"], "t_v1":  best["t1"],
                "a_v2":      a2,         "t_v2":  t2,
                "auc_v1":    best["roc1"]["auc"] if best["roc1"] else None,
                "auc_v2":    roc2["auc"]          if roc2        else None,
                "score_v1":  _composite_score(
                    best["roc1"]["auc"] if best["roc1"] else None, perf_v1),
                "score_v2":  _composite_score(
                    roc2["auc"] if roc2 else None, perf_v2),
                "perf_v1":   perf_v1,   "perf_v2":   perf_v2,
                "pt_acc_v1": pt_acc_v1, "pt_acc_v2": pt_acc_v2,
                "paths": {
                    "channel":    channel_path,  "strahler": strahler_path,
                    "h_v1":       h_v1_path,     "h_v2":    h_v2_path,
                    "hr_v1":      hr_v1_path,    "hr_v2":   hr_v2_path,
                    "gfi_v1":     gfi1_path,      "gfi_v2": GFI20_path,
                    "flood_v1":   flood_v1_path,  "flood_v2": flood_v2_path,
                    "marg":       marg_path,
                    "wd_v1":      wd_v1_path,    "wd_v2":   wd_v2_path,
                    "hzd_v1":     hzd_v1_path,   "hzd_v2":  hzd_v2_path,
                    "channel_idx":      channel_idx_path,
                },
            }

        # ======================================================================
        # FINAL SUMMARY
        # ======================================================================
        messages.addMessage("")
        messages.addMessage("=" * 60)
        messages.addMessage("OUTPUT SUMMARY  [Channel mode: {}]".format(channel_mode_label))
        messages.addMessage("=" * 60)

        for variant in variant_list:
            r = results_summary[variant]
            p = r["paths"]
            messages.addMessage("[{}]".format(LABEL[variant]))
            messages.addMessage("  Channel mode      : {}".format(channel_mode_label))
            messages.addMessage("  Best threshold    : {}".format(r["best_ci"]))
            messages.addMessage("  Channel network   : {}".format(p["channel"]))
            if p["strahler"]:
                messages.addMessage("  Strahler order    : {}".format(p["strahler"]))
            if p.get("channel_idx"):
                messages.addMessage("  {} index      : {}".format(channel_mode_label, p["channel_idx"]))
            messages.addMessage("  HAND (base)       : {}".format(p["h_v1"]))
            messages.addMessage("  HAND (backwater)  : {}".format(p["h_v2"]))
            messages.addMessage("  hr (base)         : {}".format(p["hr_v1"]))
            messages.addMessage("  hr (backwater)    : {}".format(p["hr_v2"]))
            messages.addMessage("  MGFI (base)       : {}".format(p["gfi_v1"]))
            messages.addMessage("  MGFI (backwater)  : {}".format(p["gfi_v2"]))
            messages.addMessage("  Flood prone (base): {}".format(p["flood_v1"]))
            messages.addMessage("  Flood prone (BW)  : {}".format(p["flood_v2"]))
            if p["marg"]:
                messages.addMessage("  Marginal Area     : {}".format(p["marg"]))
            if p.get("wd_v1"):
                messages.addMessage("  Water Depth (base): {}".format(p["wd_v1"]))
            if p.get("wd_v2"):
                messages.addMessage("  Water Depth (BW)  : {}".format(p["wd_v2"]))
            if p.get("hzd_v1"):
                messages.addMessage("  Hazard class v1   : {}".format(p["hzd_v1"]))
            if p.get("hzd_v2"):
                messages.addMessage("  Hazard class v2   : {}".format(p["hzd_v2"]))

            if do_calibrate:
                p1 = r["perf_v1"] or {}
                p2 = r["perf_v2"] or {}
                def _fmt(v, spec=".4f"):
                    try:
                        return format(float(v), spec)
                    except (TypeError, ValueError):
                        return str(v)
                messages.addMessage(
                    "  v1: AUC={}  Accuracy={}  Recall={}  Precision={}  F1={}  "
                    "Score={}  thr={}  a={}".format(
                        _fmt(r["auc_v1"] or 0), _fmt(p1.get("Accuracy")),
                        _fmt(p1.get("Recall (TPR)")), _fmt(p1.get("Precision")),
                        _fmt(p1.get("F1-score")), _fmt(r["score_v1"]),
                        _fmt(r["t_v1"]), _fmt(r["a_v1"], ".6f"))
                )
                messages.addMessage(
                    "  v2: AUC={}  Accuracy={}  Recall={}  Precision={}  F1={}  "
                    "Score={}  thr={}  a={}".format(
                        _fmt(r["auc_v2"] or 0), _fmt(p2.get("Accuracy")),
                        _fmt(p2.get("Recall (TPR)")), _fmt(p2.get("Precision")),
                        _fmt(p2.get("F1-score")), _fmt(r["score_v2"]),
                        _fmt(r["t_v2"]), _fmt(r["a_v2"], ".6f"))
                )
                if r["pt_acc_v1"] is not None:
                    messages.addMessage(
                        "  Point accuracy: v1={:.2f}%  v2={:.2f}%".format(
                            r["pt_acc_v1"] * 100, (r["pt_acc_v2"] or 0) * 100)
                    )
            else:
                messages.addMessage(
                    "  Manual threshold: v1={:.4f}  v2={:.4f}".format(r["t_v1"], r["t_v2"])
                )
            messages.addMessage("")

        if len(variant_list) > 1 and do_calibrate:
            messages.addMessage("=" * 60)
            messages.addMessage(
                "BEST METHOD RECOMMENDATION "
                "(ranked by composite score: AUC, Accuracy, Recall, Precision, F1-score)"
            )

            def _fmtb(v, spec=".4f"):
                try:
                    return format(float(v), spec)
                except (TypeError, ValueError):
                    return str(v)

            # Ranking table for all variants, sorted by composite score (MGFI Backwater)
            messages.addMessage(
                "  {:<32} {:>8} {:>9} {:>8} {:>9} {:>8} {:>8}".format(
                    "Method", "AUC", "Accuracy", "Recall", "Precision", "F1", "Score")
            )
            ranked_variants = sorted(
                variant_list, key=lambda v: results_summary[v]["score_v2"] or 0.0, reverse=True
            )
            for v in ranked_variants:
                rv  = results_summary[v]
                pv2 = rv["perf_v2"] or {}
                marker = " << BEST" if v == ranked_variants[0] else ""
                messages.addMessage(
                    "  {:<32} {:>8} {:>9} {:>8} {:>9} {:>8} {:>8}{}".format(
                        LABEL[v],
                        _fmtb(rv["auc_v2"] or 0),
                        _fmtb(pv2.get("Accuracy")),
                        _fmtb(pv2.get("Recall (TPR)")),
                        _fmtb(pv2.get("Precision")),
                        _fmtb(pv2.get("F1-score")),
                        _fmtb(rv["score_v2"]),
                        marker,
                    )
                )
            messages.addMessage("")

            best_v = ranked_variants[0]
            rb     = results_summary[best_v]
            p2     = rb["perf_v2"] or {}
            messages.addMessage(
                ">> {}  |  Score={}  |  AUC_v2={}  |  Accuracy={}  |  Recall={}  |  "
                "Precision={}  |  F1={}  |  CSI={}  |  Kappa={}  |  thr_v2={}".format(
                    LABEL[best_v],
                    _fmtb(rb["score_v2"]),
                    _fmtb(rb["auc_v2"] or 0),
                    _fmtb(p2.get("Accuracy")),
                    _fmtb(p2.get("Recall (TPR)")),
                    _fmtb(p2.get("Precision")),
                    _fmtb(p2.get("F1-score")),
                    _fmtb(p2.get("CSI")),
                    _fmtb(p2.get("Kappa")),
                    _fmtb(rb["t_v2"]),
                )
            )
            messages.addMessage("   MGFI (BW) : {}".format(rb["paths"]["gfi_v2"]))
            messages.addMessage("   Flood (BW): {}".format(rb["paths"]["flood_v2"]))
            if rb["paths"].get("wd_v2"):
                messages.addMessage("   WD (BW)   : {}".format(rb["paths"]["wd_v2"]))
            messages.addMessage("")
            messages.addMessage(
                "NOTE: Two Horn 3x3 kernel-based channel indices are available. "
                "Channel_ASk is slope-angle sensitive (best for hilly/mountainous terrain). "
                "Channel_GFI20 uses hydraulic gradient (best for gentle/lowland terrain) "
                "and is the recommended default. "
                "Runoff weighting (SCS-CN, GA, ICL) "
                "propagates into both via variant-specific flow accumulation."
            )

        messages.addMessage("=" * 60)
        messages.addMessage("[DONE] All processing complete.")

        v0 = variant_list[0]
        parameters[24].value = results_summary[v0]["paths"]["gfi_v1"]
        parameters[25].value = results_summary[v0]["paths"]["flood_v2"]

        arcpy.CheckInExtension("Spatial")
