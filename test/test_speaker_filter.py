import unittest
from unittest.mock import patch
import numpy as np
from app import speaker_filter as sf

class SpeakerFilterTest(unittest.TestCase):
    def test_nonmatching_windows_are_not_sent(self):
        audio = np.full(sf.RATE * 3, 500, dtype=np.int16)
        profile = np.array([1., 0.], dtype=np.float32)
        with patch.object(sf, '_profile', return_value=profile), patch.object(sf, '_embedding', side_effect=[profile, np.array([0., 1.], dtype=np.float32)]):
            result = sf.filter_speech(audio)
        self.assertTrue(np.all(result[:sf.RATE * 3 // 2] == 500))
        self.assertFalse(np.any(result[sf.RATE * 3 // 2:]))

    def test_no_matching_voice_fails_closed(self):
        with patch.object(sf, '_profile', return_value=np.array([1., 0.])), patch.object(sf, '_embedding', return_value=np.array([0., 1.])):
            with self.assertRaises(sf.SpeakerFilterError): sf.filter_speech(np.zeros(sf.RATE, dtype=np.int16))

    def test_short_voice_fails_closed(self):
        with patch.object(sf, '_profile', return_value=np.array([1., 0.])):
            with self.assertRaises(sf.SpeakerFilterError): sf.filter_speech(np.zeros(100, dtype=np.int16))

if __name__ == '__main__': unittest.main()
