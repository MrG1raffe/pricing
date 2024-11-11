import unittest
import numpy as np
from utility.utility import to_numpy

from models.hjm_model.contracts import ForwardContract

class TestForwardContract(unittest.TestCase):

    def test_contracts_dates(self):
        observation_date = np.datetime64('2020-04-11')
        names = ['CAL22', 'Q224', 'Apr23']
        contracts = []
        
        for name in names:
            contracts.append(ForwardContract(name=name, observation_date=observation_date))
        
        contracts_time_to_del_start = [contract.time_to_delivery_start for contract in contracts]
        real_contracts_time_to_del_start = [1.726027397260274, 3.9753424657534246, 2.9726027397260273]

        contracts_time_to_del_end = [contract.time_to_delivery_end for contract in contracts]
        real_contracts_time_to_del_end = [2.7260273972602738, 4.2246575342465755, 3.0547945205479454]

        self.assertTrue(np.allclose(contracts_time_to_del_start, real_contracts_time_to_del_start, rtol = 0.1))
        self.assertTrue(np.allclose(contracts_time_to_del_end, real_contracts_time_to_del_end, rtol = 0.1))


if __name__ == '__main__':
    unittest.main()