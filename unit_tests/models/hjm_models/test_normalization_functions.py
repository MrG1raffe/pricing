import unittest
import numpy as np
from utility.utility import to_numpy

from models.hjm_model.function_g import FunctionG
from models.hjm_model.contracts import ForwardContract

class TestFunctionG(unittest.TestCase):

    def test_values_and_changing_points(self):
        observation_date = np.datetime64('2024-06-03')
        contracts_name  = ['Jul24', 'Aug24', 'Q324', 'Cal25', 'Q126', 'Q226']
        contracts = []
        for name in contracts_name:
            contracts.append(ForwardContract(name = name, observation_date=observation_date))
        G = FunctionG(contracts=contracts, function_shape=lambda t : np.sin(4 * to_numpy(t)))
        
        real_changing_points = np.array([0.07671233, 0.16164384, 0.24657534, 0.32876712, 0.41369863,
                                            0.49589041, 0.58082192, 0.66575342, 0.74246575, 0.82739726,
                                            0.90958904, 0.99452055, 1.07671233, 1.16164384, 1.24657534,
                                            1.32876712, 1.41369863, 1.49589041, 1.58082192, 1.66575342,
                                            1.74246575, 1.82739726, 1.90958904, 1.99452055, 2.07671233])
        real_function_values = np.array([ 0.30205662,  0.60245655,  0.83399086,  0.96747945,  0.99647423,
                                            0.91601504,  0.7299884 ,  0.46051825,  0.1708868 , -0.16720728,
                                            -0.47658275, -0.74229543, -0.91888963, -0.99783506, -0.96271997,
                                            -0.82381969, -0.58784371, -0.29516066,  0.04009162,  0.3707611 ,
                                            0.63397137,  0.85544202,  0.97684276,  0.9923094 ,  0.89919606])

        self.assertTrue(np.allclose(real_changing_points, G.changing_points, rtol=0.1))
        self.assertTrue(np.allclose(real_function_values, G.function_values, rtol=0.1))


    def test_get_contract_changing_point(self):
        observation_date = np.datetime64('2020-04-11')
        names = ['CAL22', 'Q224', 'Apr23']
        contracts = []

        for name in names:
            contracts.append(ForwardContract(name=name, observation_date=observation_date)) 

        G = FunctionG(contracts=contracts, function_shape=lambda x : np.sin(x))

        cp = G.get_contract_changing_points(names[1])
        real_cp = np.array([3.97534247, 4.05753425, 4.14246575, 4.22465753])
 
        self.assertTrue(np.allclose(cp, real_cp, rtol=0.1))

        idx = G.get_contract_indices(contract_name=names[0])

        self.assertTrue(np.allclose(G.changing_points[idx[0]:idx[-1]+2], \
                                    G.get_contract_changing_points(contract_name=names[0]),\
                                    rtol=0.1))


if __name__ == '__main__':
    unittest.main()