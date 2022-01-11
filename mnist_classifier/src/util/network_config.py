
class NetworkConfig(object):
  def __init__(self, batch_size, learning_rate, loss_func_type="",
                     activation_func_type="") -> None:
    super().__init__()
    self.learning_rate = learning_rate
    self.loss_func_type = loss_func_type
    self.activation_func_type = activation_func_type
    self.batch_size = batch_size