import torch
import pytest

from pubmlp.calibration import TemperatureScaling


class TestTemperatureScaling:
    def test_fit_changes_temperature(self):
        torch.manual_seed(0)
        logits = torch.randn(200, 1)
        labels = (torch.sigmoid(logits * 2) > 0.5).float()
        ts = TemperatureScaling()
        ts.fit(logits, labels)
        assert ts.temperature != 1.0

    def test_transform_scales_logits(self):
        ts = TemperatureScaling()
        ts.temperature = 2.0
        logits = torch.tensor([1.0, 2.0, 4.0])
        scaled = ts.transform(logits)
        assert torch.allclose(scaled, torch.tensor([0.5, 1.0, 2.0]))

    def test_serialization_roundtrip(self):
        ts = TemperatureScaling()
        ts.temperature = 1.5
        restored = TemperatureScaling.from_dict(ts.to_dict())
        assert restored.temperature == 1.5


class TestTemperatureScalingMultiLabel:
    def test_fit_multi_label(self):
        torch.manual_seed(0)
        logits = torch.randn(200, 3)
        labels = (torch.sigmoid(logits * 2) > 0.5).float()
        ts = TemperatureScaling()
        ts.fit(logits, labels)
        assert isinstance(ts.temperature, list)
        assert len(ts.temperature) == 3

    def test_transform_multi_label(self):
        ts = TemperatureScaling()
        ts.temperature = [2.0, 0.5, 1.0]
        logits = torch.tensor([[4.0, 2.0, 3.0]])
        scaled = ts.transform(logits)
        assert torch.allclose(scaled, torch.tensor([[2.0, 4.0, 3.0]]))

    def test_serialization_multi_label(self):
        ts = TemperatureScaling()
        ts.temperature = [1.5, 2.0, 0.8]
        restored = TemperatureScaling.from_dict(ts.to_dict())
        assert restored.temperature == [1.5, 2.0, 0.8]
