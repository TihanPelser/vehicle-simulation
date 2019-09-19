import numpy as np
import pandas as pd
import logging
from scipy import interpolate
from numba import jit
import pandas as pd
from typing import Optional


class LinearTyre:
    def __init__(self, debug: bool = False):
        self.a1 = 1433.05760881
        self.a2 = 1858.01553937
        self.a3 = 2421.20122893
        self.N1 = 402.21
        self.N2 = 608.22
        self.N3 = 809.325
        self.data = []
        self.debug = debug

    def reset(self, save_name: Optional[str] = None):
        if save_name is not None and len(self.data) > 0:
            params = self.parameter_history()
            params.to_csv(f"SavedData/tyredata_{save_name}.csv")

        self.data = []

    def calculate_lateral_force(self, slip_angles: np.ndarray, normal: np.ndarray, time: float) -> np.ndarray:
        """
        Cubic 2D interpolation tyre model
        :param slip_angles: Wheel side slip angles e.g. np.array([slip_front_left, slip_front_right, slip_rear_left,
                                                                    slip_rear_right])
        :param normal: Wheel normal forces e.g. np.array([Fn_front_left, Fn_front_right, Fn_rear_left, Fn_rear_right])
        :return: Lateral forces per wheel e.g. np.array([Fy_fl, Fy_fr, Fy_rl, Fy_rr])
        """

        if np.isnan(slip_angles).any():
            logging.error(f"NaN slip angle received {slip_angles}")

        signs = np.array([-1 if (f >= 0) else 1 for f in slip_angles])
        corrected_slip = np.zeros(2)

        for i, angle in enumerate(slip_angles):
            if abs(angle) > np.radians(10):
                if self.debug:
                    print(f"Slip angle cutoff exceeded {angle}")
                corrected_slip[i] = np.radians(10)
            else:
                corrected_slip[i] = abs(angle)

        xi = np.column_stack((corrected_slip, normal))
        fy = np.zeros(2)
        for i, val in enumerate(xi):
            if val[1] < self.N1:
                val[1] = self.N1
                fy[i] = self.a1 * val[0]
            elif val[1] < self.N2 and val[1] >= self.N1:
                upperFy = self.a2 * val[0]
                lowerFy = self.a1 * val[0]
                fy[i] = np.interp(val[1], [self.N1, self.N2], [lowerFy, upperFy])

            elif val[1] < self.N3 and val[1] >= self.N2:
                upperFy = self.a3 * val[0]
                lowerFy = self.a2 * val[0]
                fy[i] = np.interp(val[1], [self.N2, self.N3], [lowerFy, upperFy])
            elif val[1] >= self.N3:
                val[1] = self.N3
                fy[i] = self.a3 * val[0]

        if np.isnan(fy).any():
            logging.error(f"TyreModel produced NaN for {xi}")
            print(fy)

        fy = fy * signs
        # print(f"Forces : {fy} \t || Slip : {slip_angles} \t || ")
        self.data.append([time, slip_angles[0], normal[0], fy[0], slip_angles[1], normal[1], fy[1]])

        return fy

    def parameter_history(self) -> pd.DataFrame:
        data_history = pd.DataFrame(self.data, columns=["Time", "Slip Front", "Normal Front", "Force Front",
                                                        "Slip Rear", "Normal Rear", "Force Rear", "Cutoff"])
        return data_history

