import unittest
import numpy as np
from utility.utility import to_numpy

from models.model_params import LiftedHestonParams, HistoricalParams, CosParams, SteinSteinParams
from models.lifted_heston import LiftedHeston
from models.stein_stein import SteinStein
from models.hjm_model.contracts import ForwardContract
from models.hjm_model.calibration import *

observation_date = np.datetime64('2024-05-15')
contract_names_squeeze = ["Jul24", "Aug24", "Q324", "Q424", "Q125", "CAL25", "CAL26"]
contract_names = ["Jul24", "Aug24", "Q324", "Q424", "Q125", "CAL25", "CAL25", "CAL26"]

F0s = [70.5, 73.45, 77, 99, 112, 98, 103, 88]
contracts = [ForwardContract(name=name, observation_date= observation_date, F0 = F0s[i]) \
             for i, name in enumerate(contract_names_squeeze)]

maturities = [0.11506849315068493,
                0.19726027397260273,
                0.11506849315068493,
                0.36712328767123287,
                0.6191780821917808,
                0.6191780821917808,
                0.36712328767123287,
                1.6191780821917807]

strikes = [[70.5],
            [73.45],
            [ 60,  65,  69,  73,  77,  81,  85,  91, 101,],
            [ 63,  74,  83,  91,  99, 109, 120, 136, 164,],
            [ 64,  77,  89, 100, 112, 125, 143, 167, 211,],
            [ 63,  74,  82,  90,  98, 107, 119, 136, 165,],
            [68, 77, 83, 89, 95, 103, 111, 123, 143],
            [ 50,  61,  70,  78,  88,  99, 115, 139, 184,]]

options = []
for i, name in enumerate(contract_names):
    options.append(VanillaOption(T=maturities[i], K= strikes[i], underlying_name=name))

n_hist_fact = 10

lifted_heston_params = LiftedHestonParams(
    theta= 1,
    lam = 1,
    nu =  0.8,
    rhos=np.ones(n_hist_fact)*0, 
    c=[1,0.5],
    x=[0.000001,10],
    V0= 1 ,
    model_type="log-normal",
    normalize_variance=False
)

steinstein_params = SteinSteinParams(
    theta= 0.4,
    kappa = 1 ,
    nu = 0.3,
    rhos= np.zeros(n_hist_fact), 
    X0= 0.5,
    model_type="log-normal",
    normalize_variance=False
)

sigmas = np.array([ 50.2041853 ,  74.96591548,  55.88067016,  24.57314869,
                    58.34940051, 130.81366209, 186.47683168, 157.85342161,
                    61.13256572,   3.83729881])
taus = np.array([0.00216364, 0.0060462 , 0.00953719, 0.03233006, 0.08070597,
                0.13502121, 0.20732922, 0.30599707, 0.4330478 , 1.28399099])

corr_mat = np.eye(n_hist_fact)

hist_params = HistoricalParams(sigmas=sigmas, taus=taus, corr_mat=corr_mat)

hjm_lif_heston = ForwardsHJM(hist_params=hist_params, stoch_vol_params=lifted_heston_params,
            model_constructor=LiftedHeston, contracts=contracts, vanilla_options=options)

hjm_stein_stein = ForwardsHJM(hist_params=hist_params, stoch_vol_params=steinstein_params,
            model_constructor=SteinStein, contracts=contracts, vanilla_options=options)

iv_mkt = [np.array([0.66]),
            np.array([0.62]),
            np.array([0.578 , 0.568 , 0.563 , 0.5655, 0.573 , 0.583 , 0.593 , 0.608 ,
                    0.628 ]),
            np.array([0.578 , 0.568 , 0.563 , 0.5655, 0.573 , 0.583 , 0.5955, 0.6105,
                    0.6305]),
            np.array([0.568 , 0.558 , 0.553 , 0.5555, 0.563 , 0.573 , 0.5855, 0.6005,
                    0.6205]),
            np.array([0.448 , 0.438 , 0.433 , 0.4355, 0.443 , 0.458 , 0.478 , 0.498 ,
                    0.518 ]),
            np.array([0.438 , 0.428 , 0.423 , 0.4255, 0.433 , 0.448 , 0.468 , 0.483 ,
                    0.503 ]),
            np.array([0.3565, 0.3465, 0.3415, 0.344 , 0.3515, 0.3665, 0.3865, 0.4065,
                    0.4315])]

pricing_params = CosParams(N_trunc=50, cf_timestep=0.003, scheme="exp")
method = 'cos'
smile_calib_idx = [2]
vol_level_calib_idx=[0,1,2,3,4,5,6]
bounds = [(0,20),(0,20),(0,20)]  + [(-1, 1)] * len(smile_calib_idx)

calibration_strategy = CalibrationStrategyHJM(smile_calib_idx=smile_calib_idx,
                                            vol_level_calib_idx=vol_level_calib_idx,
                                            pricing_method=method,
                                            pricing_params=pricing_params,
                                            optimiser = 'Powell')

calibration_data = CalibrationDataHJM(atm_idx=[0,0,4,4,4,4,4,4],
                                    implied_vols_market=iv_mkt)

calibration_lift_heston = CalibrationHJM(hjm = hjm_lif_heston,
                            calibration_data=calibration_data,
                            calibration_strategy=calibration_strategy)

calibration_stein_stein = CalibrationHJM(hjm = hjm_stein_stein,
                            calibration_data=calibration_data,
                            calibration_strategy=calibration_strategy)

class TestCalibration(unittest.TestCase):
   
    def test_init(self):
        self.assertTrue(calibration_lift_heston.calibration_data.implied_vols_market is not None)
        self.assertTrue(isinstance(calibration_lift_heston.calibration_data.implied_vols_market, List) \
                         and len(calibration_lift_heston.calibration_data.implied_vols_market) != 0)
        
        self.assertTrue(calibration_stein_stein.calibration_data.prices_market is not None)
        self.assertTrue(isinstance(calibration_stein_stein.calibration_data.prices_market, List) \
                         and len(calibration_stein_stein.calibration_data.prices_market) != 0)

    def test_smile_idx_to_contract_idx(self):
        self.assertTrue(calibration_stein_stein._smile_idx_to_contract_idx(6)==5)

    def test_updated_calibration_settup(self):
        calibration_lift_heston.update_calibration_setup(n_factors = 2, model_params=('theta', 'nu', 'c', 'x'))
        params = [5,0.2,1,2,1,10]
        model_params_dict = calibration_lift_heston.calibration_strategy.vector_to_params(params)
        calibration_lift_heston.hjm.update_stoch_vol_params(model_params_dict)

        self.assertTrue(np.allclose(calibration_lift_heston.hjm.models[0].c, [1,2], rtol = 0.1))
        self.assertTrue(np.allclose(calibration_lift_heston.hjm.models[1].x, [1,10], rtol = 0.1))
        self.assertTrue(np.allclose(calibration_lift_heston.hjm.models[2].nu, 0.2, rtol = 0.1))
        self.assertTrue(np.allclose(calibration_lift_heston.hjm.models[3].theta, 5, rtol = 0.1))

    def test_loss_prices(self):
        calibration_lift_heston.update_calibration_setup(n_factors = 2, model_params=("theta", "lam", "nu", "c", "rho_spot_vol"))
        model_loss_price = calibration_lift_heston.loss_prices([1,1,0.5,1,2,0.1])
        real_loss_price = 377.00098114321514

        self.assertTrue(np.allclose(model_loss_price, real_loss_price, rtol = 0.1))

        calibration_stein_stein.update_calibration_setup(n_factors = 1, model_params=("kappa", "nu",  "rhos"))
        model_loss_price = calibration_stein_stein.loss_prices([0.1,0.5]+list(0.1*np.ones(n_hist_fact)))
        real_loss_price = 23.641531058413786

        self.assertTrue(np.allclose(model_loss_price, real_loss_price, rtol = 0.1))
     
    
if __name__ == '__main__':
    unittest.main()