from src.data.synthetic import make_client_splits
from src.utils.config import DemoConfig


def test_partition_creates_non_empty_clients():
    config = DemoConfig(num_clients=5, samples_per_client=30, quick=True).normalized()
    clients = make_client_splits(config)

    assert len(clients) == config.num_clients
    assert all(client.train_y.numel() > 0 for client in clients)
    assert all(client.val_y.numel() > 0 for client in clients)


def test_non_iid_partition_has_label_skew():
    config = DemoConfig(
        num_clients=5,
        samples_per_client=80,
        partition="non_iid",
        dirichlet_alpha=0.2,
    )
    clients = make_client_splits(config)
    histograms = [tuple(client.label_histogram.values()) for client in clients]

    assert len(set(histograms)) > 1
