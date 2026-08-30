import torch

from speakerid.losses import AAMSoftmax
from speakerid.model import SpeakerEncoder


def test_encoder_and_aam_shapes():
    model = SpeakerEncoder(
        n_mels=80,
        channels=128,
        scale=8,
        embedding_dim=64,
    )
    x = torch.randn(4, 80, 300)
    z = model(x)
    assert z.shape == (4, 64)
    assert torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-4)

    head = AAMSoftmax(embedding_dim=64, num_classes=10)
    labels = torch.tensor([0, 1, 2, 3])
    logits = head(z, labels)
    assert logits.shape == (4, 10)
