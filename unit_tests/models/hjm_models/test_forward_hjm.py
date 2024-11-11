import unittest
import numpy as np
from utility.utility import to_numpy

from models.model_params import LiftedHestonParams
from models.hjm_model.forwards_HJM import *

class TestForwardHJM(unittest.TestCase):

    def test_update_sigmas(self):
        observation_date = np.datetime64('2024-05-15')
        contract_names = ["Jul24", "Aug24", "Q324", "Q424", "Q125", "CAL25", "CAL26"]
        contracts = [ForwardContract(name=name, observation_date=observation_date) for name in contract_names]
        maturities = [0.11506849315068493,
                        0.19726027397260273,
                        0.11506849315068493,
                        0.36712328767123287,
                        0.6191780821917808,
                        0.6191780821917808,
                        0.36712328767123287]
        strikes = [[70.5],
                   [73.45],
                   [ 60,  65,  69,  73,  77,  81,  85,  91, 101,],
                   [ 63,  74,  83,  91,  99, 109, 120, 136, 164,],
                   [ 64,  77,  89, 100, 112, 125, 143, 167, 211,],
                   [ 63,  74,  82,  90,  98, 107, 119, 136, 165,],
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

        sigmas = np.array([ 50.2041853 ,  74.96591548,  55.88067016,  24.57314869,
                            58.34940051, 130.81366209, 186.47683168, 157.85342161,
                            61.13256572,   3.83729881])
        taus = np.array([0.00216364, 0.0060462 , 0.00953719, 0.03233006, 0.08070597,
                        0.13502121, 0.20732922, 0.30599707, 0.4330478 , 1.28399099])
        
        corr_mat = np.array([[ 1.        , -0.93679543,  0.81533726, -0.47668129,  0.35528901,
                            -0.28672012,  0.2308198 , -0.18783241,  0.15698521, -0.05919811],
                        [-0.93679543,  1.        , -0.96280015,  0.70055762, -0.5295399 ,
                            0.4116603 , -0.3182086 ,  0.2522448 , -0.20945116,  0.09422323],
                        [ 0.81533726, -0.96280015,  1.        , -0.84389476,  0.65037236,
                            -0.50136474,  0.38165387, -0.29779797,  0.24481904, -0.1182522 ],
                        [-0.47668129,  0.70055762, -0.84389476,  1.        , -0.89766708,
                            0.73252908, -0.56978581,  0.44137545, -0.35447481,  0.16758167],
                        [ 0.35528901, -0.5295399 ,  0.65037236, -0.89766708,  1.        ,
                            -0.94406565,  0.82470044, -0.69728701,  0.59484716, -0.33688544],
                        [-0.28672012,  0.4116603 , -0.50136474,  0.73252908, -0.94406565,
                            1.        , -0.9609834 ,  0.87710645, -0.79140771,  0.50236903],
                        [ 0.2308198 , -0.3182086 ,  0.38165387, -0.56978581,  0.82470044,
                            -0.9609834 ,  1.        , -0.97370107,  0.92114679, -0.63735831],
                        [-0.18783241,  0.2522448 , -0.29779797,  0.44137545, -0.69728701,
                            0.87710645, -0.97370107,  1.        , -0.98437856,  0.74010941],
                        [ 0.15698521, -0.20945116,  0.24481904, -0.35447481,  0.59484716,
                            -0.79140771,  0.92114679, -0.98437856,  1.        , -0.82045336],
                        [-0.05919811,  0.09422323, -0.1182522 ,  0.16758167, -0.33688544,
                            0.50236903, -0.63735831,  0.74010941, -0.82045336,  1.        ]]) 
        
        hist_params = HistoricalParams(sigmas=sigmas, taus=taus, corr_mat=corr_mat)

        hjm = ForwardsHJM(hist_params=hist_params, stoch_vol_params=lifted_heston_params,
                  model_constructor=LiftedHeston, contracts=contracts, vanilla_options=options)
        
        hjm.g_function = FunctionG(contracts=contracts, function_shape = lambda x : np.sin(4*to_numpy(x)))
        hjm.update_sigmas()

        model_sigmas = hjm.models[2].sigmas(0)
        real_sigmas = np.array([[3.02143655e-27, 4.98988754e-10, 1.42575153e-06, 3.00708037e-02,
                                    2.13878399e+00, 1.45344334e+01, 3.85808857e+01, 4.82703917e+01,
                                    2.39324375e+01, 2.25006931e+00]])
        
        self.assertTrue(np.allclose(real_sigmas, model_sigmas, rtol=0.1))

    def test_update_stoch_vol_params(self):
        observation_date = np.datetime64('2024-05-15')
        contract_names = ["Jul24", "Aug24", "Q324", "Q424", "Q125", "CAL25", "CAL26"]
        contracts = [ForwardContract(name=name, observation_date=observation_date) for name in contract_names]
        maturities = [0.11506849315068493,
                        0.19726027397260273,
                        0.11506849315068493,
                        0.36712328767123287,
                        0.6191780821917808,
                        0.6191780821917808,
                        0.36712328767123287]
        strikes = [[70.5],
                   [73.45],
                   [ 60,  65,  69,  73,  77,  81,  85,  91, 101,],
                   [ 63,  74,  83,  91,  99, 109, 120, 136, 164,],
                   [ 64,  77,  89, 100, 112, 125, 143, 167, 211,],
                   [ 63,  74,  82,  90,  98, 107, 119, 136, 165,],
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

        sigmas = np.array([ 50.2041853 ,  74.96591548,  55.88067016,  24.57314869,
                            58.34940051, 130.81366209, 186.47683168, 157.85342161,
                            61.13256572,   3.83729881])
        taus = np.array([0.00216364, 0.0060462 , 0.00953719, 0.03233006, 0.08070597,
                        0.13502121, 0.20732922, 0.30599707, 0.4330478 , 1.28399099])
        
        corr_mat = np.eye(n_hist_fact)
        
        hist_params = HistoricalParams(sigmas=sigmas, taus=taus, corr_mat=corr_mat)

        hjm = ForwardsHJM(hist_params=hist_params, stoch_vol_params=lifted_heston_params,
                  model_constructor=LiftedHeston, contracts=contracts, vanilla_options=options)
        
        model_params_dict = {"theta" : 4, "c" :  [2,5], "rhos" :  np.ones(n_hist_fact)*0.3}
        hjm.update_stoch_vol_params(model_params_dict=model_params_dict)

        self.assertTrue(np.allclose(hjm.models[0].theta, 4, rtol=0.1))
        self.assertTrue(np.allclose(hjm.models[3].c, [2,5], rtol=0.1))
        self.assertTrue(np.allclose(hjm.models[4].rhos, np.ones(n_hist_fact)*0.3, rtol=0.1))

if __name__ == '__main__':
    unittest.main()