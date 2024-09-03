# Structure for training directory:
#
# $BASE_DIR_NAME[COM_DANNCE_TRAINING]/
#   calibration/
#       hires_cam1_params.mat
#       hires_cam2_params.mat
#       ...
#   COM/train01/checkpoint-final.pth
#   DANNCE/train01/checkpoint-final.pth
#   videos/
#       Camera1/0.mp4
#       Camera2/0.mp4
#       ...
#   tmp_Label3D_dannce.mat
#   tmp_COM_Label3D_dannce.mat
#   io.yaml
#   README.md


from ruamel.yaml import YAML, CommentedMap
from pathlib import Path
from datetime import datetime
import shutil
import sys


# TEST DATA:
################################################
test_dannce_exp_list = [
    {
        "label3d_file": "/n/holylabs/LABS/olveczky_lab/Lab/dannce-dev/hannah-data/Alone_COM_label3d_files_Diego/2/20201107_164529_Label3D_dannce.mat",
        "com_file": "/n/holylabs/LABS/olveczky_lab/Lab/dannce-dev/hannah-data/Alone_COM_label3d_files_Diego/2/com3d.mat",
    },
    {
        "label3d_file": "/n/holylabs/LABS/olveczky_lab/Lab/dannce-dev/hannah-data/Alone_COM_label3d_files_Diego/3/total_dannce.mat",
        "com_file": "/n/holylabs/LABS/olveczky_lab/Lab/dannce-dev/hannah-data/Alone_COM_label3d_files_Diego/3/com3d.mat",
    },
    {
        "label3d_file": "/n/holylabs/LABS/olveczky_lab/Lab/dannce-dev/hannah-data/Alone_COM_label3d_files_Diego/5/total_dannce.mat",
        "com_file": "/n/holylabs/LABS/olveczky_lab/Lab/dannce-dev/hannah-data/Alone_COM_label3d_files_Diego/5/com3d.mat",
    },
]

test_com_exp_list = [
    {
        "label3d_file": "/n/olveczky_lab_tier1/Lab/dannce_rig2/data/M1-M7_photometry/Alone/Day1_wk2/240624_135840_M4/20240808_125105_COM_Label3D_dannce.mat"
    },
    {
        "label3d_file": "/n/olveczky_lab_tier1/Lab/dannce_rig2/data/M1-M7_photometry/Alone/Day2_wk2/240625_143814_M5/20240808_125331_COM_Label3D_dannce.mat"
    },
    {
        "label3d_file": "/n/olveczky_lab_tier1/Lab/dannce_rig2/data/M1-M7_photometry/Alone/Day2_wk2/240625_160541_M7/20240808_125455_COM_Label3D_dannce.mat"
    },
    {
        "label3d_file": "/n/olveczky_lab_tier1/Lab/dannce_rig2/data/M1-M7_photometry/Alone/Day3_wk2/240626_114520_M5/20240808_125726_COM_Label3D_dannce.mat"
    },
]

########################################################################


_TRAINING_DIRS = ["calibration", "COM", "DANNCE", "videos"]
_TRAINING_EMPTY_FILES = ["./tmp_empty_dannce.mat"]
N_CAMERAS = 6


def make_io_yaml_data(com_exp_list, dannce_exp_list):
    """Returns a map representation of io.yaml.data"""
    data = CommentedMap(
        {
            "com_train_dir": "./COM/train01",
            "com_predict_dir": None,
            "com_predict_weights": None,
            "com_exp": com_exp_list,
            "dannce_train_dir": "./DANNCE/train01",
            "dannce_predict_weights": None,
            "dannce_predict_model": None,
            "exp": dannce_exp_list,
        }
    )

    # add useful comments to the yaml file
    data.yaml_set_start_comment(
        '### COM_DANNCE_TRAINING folder serves as the "project folder" for COM & DANNCE training'
        + f"\n### Auto-generated by DANNCE_GUI script on {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )
    data.yaml_set_comment_before_after_key(
        key="com_train_dir", before="\nCOM EXPERIMENTS AND OUTPUT DIRECTORIES"
    )
    data.yaml_set_comment_before_after_key(
        key="dannce_train_dir", before="\nDANNCE EXPERIMENTS AND OUTPUT DIRECTORIES"
    )
    data.yaml_add_eol_comment(
        "Not used for training project folder", key="com_predict_dir", column=30
    )
    data.yaml_add_eol_comment(
        "Not used for training project folder", key="com_predict_weights"
    )
    data.yaml_add_eol_comment(
        "Not used for training project folder", key="dannce_predict_dir", column=30
    )
    data.yaml_add_eol_comment(
        "Not used for training project folder", key="dannce_predict_model"
    )
    return data


def make_training_dir(
    base_dir: str = "./TMP_TRAIN_FOLDER",
    delete_if_exists=False,
    com_exp_list: list[dict] = [],
    dannce_exp_list: list[dict] = [],
):
    """Create a directory tree at $base_dir for holding COM and DANNCE training"""

    base_dir: Path = Path(base_dir)

    if delete_if_exists:
        # delete and re-create the directory if it already exists
        if base_dir.is_dir():
            shutil.rmtree(base_dir)

    if base_dir.is_dir():
        raise Exception(
            f"Training Dir already exists ({str(base_dir)}). Either delete directory or set arg delete_if_exists=True"
        )

    base_dir.mkdir()

    for s in _TRAINING_DIRS:
        p = Path(base_dir, s)
        p.mkdir(parents=True)

    for s in _TRAINING_EMPTY_FILES:
        p = Path(base_dir, s)
        p.touch()

    for cam_idx in range(N_CAMERAS):
        camera_name = f"Camera{cam_idx+1}"
        p_dir = Path(base_dir, "videos", camera_name)
        p_dir.mkdir(parents=True)
        p_vid = Path(p_dir, "0.mp4")
        p_vid.touch()

    file_io_yaml = Path(base_dir, "io.yaml")

    data_io_yaml = make_io_yaml_data(
        com_exp_list=com_exp_list, dannce_exp_list=dannce_exp_list
    )
    yaml = YAML(typ="rt")
    yaml.width = 10000  # prevent line-wrapping for long lines

    with open(file_io_yaml, "wt") as f:
        yaml.dump(data_io_yaml, f)

    return str(base_dir)


if __name__ == "__main__":
    dest = sys.argv[1]
    if not dest:
        raise Exception("Must dest folder location")
    print("CREATING FOLDER AT DEST: ", dest)
    make_training_dir(
        base_dir=dest,
        delete_if_exists=True,
        com_exp_list=test_com_exp_list,
        dannce_exp_list=test_dannce_exp_list,
    )
