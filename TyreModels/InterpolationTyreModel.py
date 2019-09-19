import numpy as np
import pandas as pd
import logging
from scipy import interpolate
from numba import jit
import pandas as pd
from typing import Optional

class TyreModel:
    """
    Tyre Model which provides fuctions to calculate the generated lateral force, given a slip angle and normal load
    """

    xy: np.ndarray
    z: np.array

    def __init__(self, debug: bool = False):
        try:
            data = pd.read_csv(f"./TyreModels/Tyre_Model.txt", sep="\t").values
            self.xy = np.stack((np.radians(data[:, 0]), data[:, 2]), axis=1)
            self.z = data[:, 1]
            self.exceeded_slip_angle_max_count = 0
            self.data = []
            self.cutoff = False
            self.interpolate = interpolate.CloughTocher2DInterpolator(points=self.xy, values=self.z)
        except FileNotFoundError:
            logging.error("Tyre model data file not found. Check path in config file")
            exit(1)
        except KeyError as e:
            logging.error("Tyre config file missing/incorrect")
            logging.error(e)
            exit(1)

    def reset(self, save_name: Optional[str] = None):

        if save_name is not None and len(self.data) > 0:
            params = self.parameter_history()
            params.to_csv(f"SavedData/tyredata_{save_name}.csv")

        self.exceeded_slip_angle_max_count = 0
        self.data = []
        self.cutoff = False

    def calculate_lateral_force(self, slip_angles: np.ndarray, normal: np.ndarray, time: float) -> np.ndarray:
        """
        Cubic 2D interpolation tyre model
        :param slip_angles: Wheel side slip angles e.g. np.array([slip_front_left, slip_front_right, slip_rear_left,
                                                                    slip_rear_right])
        :param normal: Wheel normal forces e.g. np.array([Fn_front_left, Fn_front_right, Fn_rear_left, Fn_rear_right])
        :return: Lateral forces per wheel e.g. np.array([Fy_fl, Fy_fr, Fy_rl, Fy_rr])
        """
        self.cutoff = False
        signs = np.array([-1 if (f >= 0) else 1 for f in slip_angles])
        corrected_slip = np.zeros(2)

        for i, angle in enumerate(slip_angles):
            if abs(angle) > np.radians(19):
                if self.debug:
                    print(f"Slip angle exceeded {angle}")
                corrected_slip[i] = np.radians(19)
                self.exceeded_slip_angle_max_count += 1
                self.cutoff = True
            else:
                corrected_slip[i] = abs(angle)

        xi = np.column_stack((corrected_slip, normal))

        for index, value in enumerate(xi):
            if value[0] > np.radians(19):
                print(f"Slip angle exceeded {value}")
                xi[index][0] = np.radians(19)
                self.exceeded_slip_angle_max_count += 1
                self.cutoff = True

        fy = self.interpolate(xi)

        if np.isnan(fy).any():
            logging.error(f"TyreModel produced NaN for {xi}")
            print(fy)

        fy = fy * signs
        # print(f"Forces : {fy} \t || Slip : {slip_angles} \t || ")
        self.data.append([time, slip_angles[0], normal[0], fy[0], slip_angles[1], normal[1], fy[1], self.cutoff])

        return fy

    def cornering_stiffness(self, axle: int) -> float:
        # TODO: Calculate cornering stiffness
        """
        :param axle: The axle for which the cornering stiffness should be calculated (Front = 1, Rear = 0)
        :return: Cornering stiffness of the wheels on the specified axle (Float)
        """
        if axle == 1:
            return 1.5
        else:
            return 1.

    def parameter_history(self) -> pd.DataFrame:
        data_history = pd.DataFrame(self.data, columns=["Time", "Slip Front", "Normal Front", "Force Front",
                                                        "Slip Rear", "Normal Rear", "Force Rear", "Cutoff"])
        return data_history
