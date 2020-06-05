import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import unittest

from onnx import numpy_helper


class AtomicModel(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(AtomicModel, self).__init__()

        self.conv = nn.Conv2d(kernel_size=kernel_size, 
                              in_channels=in_channels,
                              out_channels=out_channels)
    def forward(self, x):
        return self.conv(x)

class TestIOBinding(unittest.TestCase):

    def test_bind_input_only(self):
        kernel_size = 5
        channels = 16
        device = torch.device('cuda')
        batch_size = 1
        sample_dim = 256

        model = AtomicModel(kernel_size=kernel_size, in_channels=channels, out_channels=channels)
        model.eval()
        model.to(device)
        
        # Run Pytorch
        x = torch.randn(batch_size, channels, sample_dim, sample_dim).to(device)
        torch_output = model(x).cpu().detach().numpy()
        
        # Run ORT
        input_names = [ "input" ]
        output_names = [ "output" ]
        torch.onnx.export(model, x, "model.onnx", 
                          input_names=input_names, output_names=output_names,
                          dynamic_axes={"input":{0:"batch_size"}, "output":{0:"batch_size"}})
        
        session = onnxruntime.InferenceSession('model.onnx')

        io_binding = session.io_binding()
        io_binding.bind_input('input', x.device.type, 0, np.float32, list(x.size()), x.data_ptr())
        io_binding.bind_output('output')
        session.run_with_iobinding(io_binding)
        ort_output = io_binding.get_outputs()[0]
    
        self.assertTrue(np.array_equal(torch_output, ort_output))
    
if __name__ == '__main__':
    unittest.main()
