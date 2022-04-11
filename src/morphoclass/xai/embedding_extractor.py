"""Implementation of the `EmbeddingExtractor` class."""
from __future__ import annotations


class EmbeddingExtractor:
    """Extract inputs/outputs from a given layer.

    An instance of the embedding extractor is created by providing a model
    and a layer in that model for which the inputs and the outputs should
    be extracted.

    To obtain the inputs and the outputs call the extractor's `__call__`
    function with a batch of data, which is internally forwarded to the model.
    the `__call__` function then returns the inputs/outputs as a tuple.

    Parameters
    ----------
    model : torch.nn.Module
        A torch model from which to extract intermediate activations.
    layer_name : str
        The layer in the given model for which to extract the inputs/outputs.
        The layer instance is obtained by `getattr(model, layer_name)` and
        should return an instance of `torch.nn.Module`.
    """

    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self._last_input = None
        self._last_output = None
        self._hook_handle = None

        self._install_hook()

    def _install_hook(self):
        def forward_hook(module, layer_input, layer_output):
            if not isinstance(layer_input, tuple):
                layer_input = (layer_input,)
            if not isinstance(layer_output, tuple):
                layer_output = (layer_output,)
            self._last_input = tuple(x.detach().cpu().numpy() for x in layer_input)
            self._last_output = tuple(x.detach().cpu().numpy() for x in layer_output)

        layer = getattr(self.model, self.layer_name)
        layer._forward_hooks.clear()
        self._hook_handle = layer.register_forward_hook(forward_hook)

    def __call__(self, batch):
        """Run forward pass on the batch and get the embedding data.

        Parameters
        ----------
        batch
            A data batch suitable for the forward pass of the model.

        Returns
        -------
        torch.Tensor
            The last input data of the embedding layer.
        torch.Tensor
            The last output data of the embedding layer.
        """
        device = next(self.model.parameters()).device
        self.model(batch.to(device))

        return self._last_input, self._last_output

    def __del__(self):
        """Delete the class instance."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
