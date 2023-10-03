import pyotb

labels_img = "/data/terrain_truth/amsterdam_labelimage.tif"
vec_train = "/data/output/vec_train.geojson"
vec_valid = "/data/output/vec_valid.geojson"
vec_test = "/data/output/vec_test.geojson"

pyotb.PatchesSelection({
    "in": labels_img,
    "grid.step": 64,
    "grid.psize": 64,
    "strategy": "split",
    "strategy.split.trainprop": 0.80,
    "strategy.split.validprop": 0.10,
    "strategy.split.testprop": 0.10,
    "outtrain": vec_train,
    "outvalid": vec_valid,
    "outtest": vec_test
})

import os
os.environ["OTB_TF_NSOURCES"] = "3"

p_img =  ("/data/spot/PROD_SPOT7_001/VOL_SPOT7_001_A/IMG_SPOT7_P_001_A/"
          "DIM_SPOT7_P_201409171025192_ORT_1190912101.XML")
xs_img = ("/data/spot/PROD_SPOT7_001/VOL_SPOT7_001_A/IMG_SPOT7_MS_001_A/"
          "DIM_SPOT7_MS_201409171025192_ORT_1190912101.XML")
out_pth = "/data/output/"

for vec in [vec_train, vec_valid, vec_test]:
    app_extract = pyotb.PatchesExtraction({
        "source1.il": p_img,
        "source1.patchsizex": 64,
        "source1.patchsizey": 64,
        "source1.nodata": 0,
        "source2.il": xs_img,
        "source2.patchsizex": 16,
        "source2.patchsizey": 16,
        "source2.nodata": 0,
        "source3.il": labels_img,
        "source3.patchsizex": 64,
        "source3.patchsizey": 64,
        "vec": vec,
        "field": "id"
    })
    name = vec.replace("vec_", "").replace(".geojson", "")
    out_dict = {
        "source1.out": name + "_p_patches.tif",
        "source2.out": name + "_xs_patches.tif",
        "source3.out": name + "_labels_patches.tif",
    }
    pixel_type = {
        "source1.out": "int16",
        "source2.out": "int16",
        "source3.out": "uint8",
    }
    ext_fname = "gdal:co:COMPRESS=DEFLATE"
    app_extract.write(out_dict, pixel_type=pixel_type, ext_fname=ext_fname)
