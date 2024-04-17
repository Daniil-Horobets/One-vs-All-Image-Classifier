import torch
from models.combined_model import CombinedModel

'''
    .. Note: models in the constructor should be the same as the ones used in training. E.g. if you trained with 
        crack_model=CustomResNet50Model and inactive_model=CustomResNet18Model same models should be passed to the
        constructor.
'''
class Exporter:
      def __init__(self, model_crack, model_inactive):
            self.checkpoints_dir = "checkpoints/"
            self.model_crack = model_crack
            self.model_inactive = model_inactive

      def export(self, model_crack_name, model_inactive_name, output_name):
            # Load the models
            self.model_crack.load_state_dict(torch.load(self.checkpoints_dir + model_crack_name))
            self.model_inactive.load_state_dict(torch.load(self.checkpoints_dir + model_inactive_name))

            # Combine models into a single model with two outputs
            combined_model = CombinedModel(self.model_crack, self.model_inactive)

            fn = self.checkpoints_dir + output_name
            m = combined_model.cpu()
            m.eval()
            x = torch.randn(1, 3, 300, 300, requires_grad=True)
            y = combined_model(x)

            torch.onnx.export(m,             # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  fn,                        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
