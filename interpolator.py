# interpolator.py
#   @ description:
#       Implement of several interpolation methods for ur5e_driver. The goal
#       is to interpolate the curve as smooth as you can to elimate shifting
#   
#   @ Implemented interpolator
#       1. Linear
#       2. Cubic

import numpy as np
from typing import List, Union
from scipy.interpolate import interp1d, \
        CubicHermiteSpline, \
        CubicSpline


class OnlineLinearInterpolator:
    DOF_VECTOR = np.array
    def __init__(self, max_points: int = 20, dof: int = 6):
        self.max_points = max_points
        self.dof = dof
        self.x_data: List[float] = []
        self.y_data: List[DOF_VECTOR] = []
        # The type of interp_funcs will now be a list of interp1d functions
        self.interp_funcs: Union[List[interp1d], None] = None

    def add_point(self, x: float, y: DOF_VECTOR):
        y = np.asarray(y)
        if y.shape != (self.dof,):
            raise ValueError(f"Expected shape ({self.dof},), got {y.shape}")

        self.x_data.append(float(x))
        self.y_data.append(y)

        if len(self.x_data) > self.max_points:
            self.x_data = self.x_data[-self.max_points:]
            self.y_data = self.y_data[-self.max_points:]

        self._fit()

    def _fit(self):
        if len(self.x_data) < 2: # Linear interpolation requires at least 2 points
            self.interp_funcs = None
            return

        x = np.array(self.x_data)
        y_arr = np.array(self.y_data)  # shape (n, dof)
        funcs = []
        for i in range(self.dof):
            funcs.append(interp1d(x, y_arr[:, i], kind='linear', \
                    fill_value='extrapolate'))
        self.interp_funcs = funcs

    def is_ready(self) -> bool:
        return self.interp_funcs is not None and len(self.x_data) > 1

    def evaluate(self, x_eval: float) -> Union[DOF_VECTOR, None]:
        if not self.interp_funcs:
            return None
            
        if x_eval + 5e-2 > self.x_data[-1]:
            # Return the last data point if the evaluation time is slightly ahead
            return self.y_data[-1] 
            
        if x_eval < self.x_data[0]:
            return self.y_data[0]

        result = np.zeros(self.dof)
        for i, f in enumerate(self.interp_funcs):
            result[i] = f(x_eval)
        return result


class OnlineCubicInterpolator:
    def __init__(self, max_point: int = 20, dof: int = 6):
        self.max_points = max_point
        self.dof = dof
        self.x_data: List[float] = []
        self.y_data: List[np.array] = []
        self.interp_args: List[np.array] = []


    @staticmethod
    def interpolate_cubic(y0: np.array, dy0: np.array, \
            y1: np.array, dy1: np.array) -> tuple:
        d = y0
        c = dy0
        b = 3 * (y1 - y0) - (2 * dy0 + dy1)
        a = 2 * (y0 - y1) + (dy0 + dy1)
        return a, b, c, d


    @staticmethod
    def evaluate_polynomial(x: np.array, \
            a: np.array, b: np.array, c: np.array, d: np.array) -> np.array:
        res = a * x**3 + b * x**2 + c * x + d
        return res


    def add_point(self, x: float, y: np.array):
        y = np.asarray(y)
        self.x_data.append(float(x))
        self.y_data.append(y)
        if len(self.x_data) >= 3:
            if x - self.x_data[-2] < 1e-3:
                raise Exception("Frequency too high for this interpolation!")
            y0 = self.y_data[-2]
            dy0 = (self.y_data[-2] - self.y_data[-3]) \
                    * (self.x_data[-1] - self.x_data[-2]) \
                    / (self.x_data[-2] - self.x_data[-3])
            y1 = self.y_data[-1]
            dy1 = (self.y_data[-1] - self.y_data[-2])
            # a, b, c, d are all vectors
            a, b, c, d = OnlineCubicInterpolator.interpolate_cubic(\
                    y0, dy0, y1, dy1)
        else:
            a = np.zeros(self.dof)
            b, c, d = a, a, a
        self.interp_args.append((a, b, c, d))
        if len(self.x_data) > self.max_points:
            self.x_data = self.x_data[-self.max_points:]
            self.y_data = self.y_data[-self.max_points:]
            self.interp_args = self.interp_args[-self.max_points:]

    
    def is_ready(self) -> bool:
        return len(self.x_data) > 6


    def evaluate(self, x_eval: float) -> np.array:
        pos = np.searchsorted(self.x_data, x_eval) # pos is the first one >= x
        if pos == 0: # before the begin of the saved data
            return self.y_data[0]
        if pos == len(self.x_data): # exceed the end of the saved data
            return self.y_data[-1]
        # notice that the interp.ed cubic poly for segment [xi, xi+1] is 
        # stored in index i+1, but the variables have substracted xi
        a, b, c, d = self.interp_args[pos] 
        x = (x_eval - self.x_data[pos - 1]) \
                / (self.x_data[pos] - self.x_data[pos-1])
        res = OnlineCubicInterpolator.evaluate_polynomial(x, a, b, c, d)
        if np.max(np.abs(res)) > 5:
            print(f"ERROR")
            print(f"a, b, c, d = \n   {a}\n   {b}\n   {c}\n   {d}")
            print(f"x_data: {self.x_data}")
            print(f"y_data: {self.y_data}")
            print(f"eval = {x}")
            print(f"res = {res}")
            print(f"pos = {pos}")
        return res

        
class OnlineHermiteSplineInterpolator:
    """
    Online Hermite interpolator supporting incremental addition of points.

    Each degree of freedom (DOF) is interpolated independently using
    cubic Hermite splines, which ensure C1 continuity (smooth position
    and velocity).  Slopes are estimated automatically from data.
    """
    DOF_VECTOR = np.array

    def __init__(self, max_points: int = 20, dof: int = 6):
        self.max_points = max_points
        self.dof = dof
        self.x_data: List[float] = []
        self.y_data: List[np.ndarray] = []
        self.y_dot_data: List[np.ndarray] = []  # estimated derivatives
        self.interp_funcs: Union[List[CubicHermiteSpline], None] = None

    # ---------------------------------------------------------
    def add_point(self, x: float, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        if y.shape != (self.dof,):
            raise ValueError(f"Expected shape ({self.dof},), got {y.shape}")

        self.x_data.append(float(x))
        self.y_data.append(y)

        # keep at most max_points
        if len(self.x_data) > self.max_points:
            self.x_data = self.x_data[-self.max_points:]
            self.y_data = self.y_data[-self.max_points:]

        # update derivative estimates and fit splines
        self._fit()

    # ---------------------------------------------------------
    def _estimate_derivatives(self) -> np.ndarray:
        """Estimate first derivatives (slopes) for Hermite interpolation."""
        x = np.array(self.x_data)
        y = np.array(self.y_data)  # shape (n, dof)
        n = len(x)

        if n < 2:
            return np.zeros_like(y)

        dydx = np.zeros_like(y)

        # Central difference for internal points
        for i in range(1, n - 1):
            dx1 = x[i] - x[i - 1]
            dx2 = x[i + 1] - x[i]
            dydx[i] = ((y[i + 1] - y[i]) / dx2 * dx1 +
                       (y[i] - y[i - 1]) / dx1 * dx2) / (dx1 + dx2)

        # Forward / backward for boundaries
        dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
        dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return dydx

    # ---------------------------------------------------------
    def _fit(self):
        if len(self.x_data) < 2:
            self.interp_funcs = None
            return

        x = np.array(self.x_data)
        y_arr = np.array(self.y_data)
        ydot_arr = self._estimate_derivatives()
        funcs = []

        for i in range(self.dof):
            funcs.append(CubicHermiteSpline(x, y_arr[:, i], ydot_arr[:, i]))
        self.interp_funcs = funcs

    # ---------------------------------------------------------
    def is_ready(self) -> bool:
        return self.interp_funcs is not None and len(self.x_data) > 1

    # ---------------------------------------------------------
    def evaluate(self, x_eval: float) -> Union[DOF_VECTOR, None]:
        if not self.is_ready():
            return None

        # Clamp extrapolation
        if x_eval + 5e-2 > self.x_data[-1]:
            return self.y_data[-1]
        if x_eval < self.x_data[0]:
            return self.y_data[0]

        result = np.zeros(self.dof)
        for i, f in enumerate(self.interp_funcs):
            result[i] = f(x_eval)
        return result


class OnlineCubicSplineInterpolator:
    """
    Online cubic spline interpolator supporting incremental point addition.

    Ensures C2 continuity across all segments (smooth position, velocity,
    and acceleration). Each DOF is treated independently.
    """

    DOF_VECTOR = np.array

    def __init__(self, max_points: int = 20, dof: int = 6, bc_type: str = 'natural'):
        """
        Parameters
        ----------
        max_points : int
            Maximum number of stored data points.
        dof : int
            Degrees of freedom (dimensions per sample).
        bc_type : str or 2-tuple
            Boundary condition type passed to scipy.interpolate.CubicSpline:
            - 'natural' (default): zero second derivative at ends
            - 'clamped': zero slope at ends
            - ((1, dy0), (1, dyN)): custom slope boundaries
        """
        self.max_points = max_points
        self.dof = dof
        self.bc_type = bc_type

        self.x_data: List[float] = []
        self.y_data: List[np.ndarray] = []
        self.interp_funcs: Union[List[CubicSpline], None] = None

    # ---------------------------------------------------------
    def add_point(self, x: float, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        if y.shape != (self.dof,):
            raise ValueError(f"Expected shape ({self.dof},), got {y.shape}")

        self.x_data.append(float(x))
        self.y_data.append(y)

        if len(self.x_data) > self.max_points:
            self.x_data = self.x_data[-self.max_points:]
            self.y_data = self.y_data[-self.max_points:]

        self._fit()

    # ---------------------------------------------------------
    def _fit(self):
        """Recompute cubic spline interpolators for all DOFs."""
        if len(self.x_data) < 2:
            self.interp_funcs = None
            return

        x = np.array(self.x_data)
        y_arr = np.array(self.y_data)  # shape (n, dof)
        funcs = []

        for i in range(self.dof):
            funcs.append(CubicSpline(x, y_arr[:, i], bc_type=self.bc_type, extrapolate=True))
        self.interp_funcs = funcs

    # ---------------------------------------------------------
    def is_ready(self) -> bool:
        return self.interp_funcs is not None and len(self.x_data) > 1

    # ---------------------------------------------------------
    def evaluate(self, x_eval: float) -> Union[DOF_VECTOR, None]:
        """Evaluate interpolated position."""
        if not self.is_ready():
            return None

        if x_eval + 5e-2 > self.x_data[-1]:
            return self.y_data[-1]
        if x_eval < self.x_data[0]:
            return self.y_data[0]

        result = np.zeros(self.dof)
        for i, f in enumerate(self.interp_funcs):
            result[i] = f(x_eval)
        return result

    # ---------------------------------------------------------
    def evaluate_with_derivative(self, x_eval: float):
        """
        Evaluate both position and first derivative (velocity).
        Returns (y, ydot).
        """
        if not self.is_ready():
            return None, None

        if x_eval + 5e-2 > self.x_data[-1]:
            return self.y_data[-1], np.zeros(self.dof)
        if x_eval < self.x_data[0]:
            return self.y_data[0], np.zeros(self.dof)

        y = np.zeros(self.dof)
        ydot = np.zeros(self.dof)
        for i, f in enumerate(self.interp_funcs):
            y[i] = f(x_eval)
            ydot[i] = f(x_eval, 1)  # 1st derivative
        return y, ydot
