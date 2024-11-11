import unittest
import numpy as np

from pricing.utility import *


class TestUtility(unittest.TestCase):

    def test_is_call(self):
        self.assertTrue(is_call('c'))
        self.assertTrue(is_call('C'))
        self.assertTrue(is_call('call'))
        self.assertTrue(is_call('Call'))
        self.assertFalse(is_call('p'))
        self.assertFalse(is_call('P'))
        self.assertFalse(is_call('put'))
        self.assertFalse(is_call('Put'))
        self.assertFalse(is_call('Swing?!'))
        self.assertFalse(is_call('Asian'))

    def test_is_put(self):
        self.assertFalse(is_put('c'))
        self.assertFalse(is_put('C'))
        self.assertFalse(is_put('call'))
        self.assertFalse(is_put('Call'))
        self.assertTrue(is_put('p'))
        self.assertTrue(is_put('P'))
        self.assertTrue(is_put('put'))
        self.assertTrue(is_put('Put'))
        self.assertFalse(is_put('Swing?!'))
        self.assertFalse(is_put('Bermudan'))


if __name__ == '__main__':
    unittest.main()