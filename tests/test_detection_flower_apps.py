import pytest
from flwr.app import Context, RecordDict
from PIL import Image

from src.data.detection_data import load_detection_bundle
from src.data.detection_manifest import (
    collect_detection_samples,
    generate_detection_manifest_rows,
    write_detection_manifest,
)
from src.fl.detection_clientapp import (
    detection_config_from_context,
    select_detection_client,
)
import src.fl.detection_serverapp as serverapp
from src.fl.detection_serverapp import _build_strategy, _round_config
from src.utils.detection_config import DetectionConfig


def _supported_env_names():
    base = (
        "RUN_ID",
        "OUTPUT_DIR",
        "MANIFEST_PATH",
        "DATA_ROOT",
        "CLIENT_ID",
        "NUM_CLIENTS",
        "NUM_CLASSES",
        "IMAGE_SIZE",
        "BATCH_SIZE",
        "LOCAL_EPOCHS",
        "CENTRALIZED_EPOCHS",
        "NUM_ROUNDS",
        "LR",
        "MOMENTUM",
        "WEIGHT_DECAY",
        "NUM_WORKERS",
        "SCORE_THRESHOLD",
        "DEVICE",
        "PRETRAINED",
        "SEED",
    )
    legacy = {
        "RUN_ID": "EXP_ID",
        "DATA_ROOT": "ROOT_DIR",
    }
    names = []
    for suffix in base:
        names.append(f"FL_{suffix}")
        names.append(f"FL_DET_{legacy.get(suffix, suffix)}")
    return names


@pytest.fixture(autouse=True)
def _isolate_env_from_local_dotenv(tmp_path, monkeypatch):
    import src.utils.env as env_module

    monkeypatch.setattr(env_module, "DEFAULT_ENV_PATH", tmp_path / "missing.env")
    for name in _supported_env_names():
        monkeypatch.delenv(name, raising=False)


def _context(run_config=None, node_config=None, node_id=0):
    return Context(
        run_id=1,
        node_id=node_id,
        node_config=node_config or {},
        state=RecordDict(),
        run_config=run_config or {},
    )


def _make_bundle(tmp_path, sites=("site-a", "site-b")):
    (tmp_path / "images").mkdir()
    (tmp_path / "voc_labels").mkdir()
    spec = {f"h{i}": ["helmet"] for i in range(4)}
    spec.update({f"v{i}": ["safety-vest"] for i in range(4)})
    for stem, classes in spec.items():
        Image.new("RGB", (48, 48), color=(20, 50, 70)).save(
            tmp_path / "images" / f"{stem}.png"
        )
        objects = "".join(
            f"<object><name>{name}</name><bndbox>"
            f"<xmin>2</xmin><ymin>2</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object>"
            for name in classes
        )
        (tmp_path / "voc_labels" / f"{stem}.xml").write_text(
            f"<annotation>{objects}</annotation>", encoding="utf-8"
        )
    samples = collect_detection_samples(tmp_path / "voc_labels", tmp_path / "images")
    rows = generate_detection_manifest_rows(
        samples,
        sites=list(sites),
        per_site=4,
        val_fraction=0.5,
        seed=2026,
    )
    manifest = tmp_path / "manifest.csv"
    write_detection_manifest(rows, manifest)
    return load_detection_bundle(manifest, tmp_path, image_size=64), manifest


def test_client_selection_uses_single_shard_without_partition(tmp_path):
    bundle, _ = _make_bundle(tmp_path, sites=("site-a",))
    client = select_detection_client(_context(node_id=99), bundle)
    assert client.client_label == "site-a"


def test_client_selection_supports_explicit_client_id(tmp_path):
    bundle, _ = _make_bundle(tmp_path)
    client = select_detection_client(_context(node_config={"client-id": "site-b"}), bundle)
    assert client.client_label == "site-b"


def test_client_selection_falls_back_to_partition_id(tmp_path):
    bundle, _ = _make_bundle(tmp_path)
    client = select_detection_client(_context(node_config={"partition-id": 3}), bundle)
    assert client.client_label == "site-b"


def test_node_config_overrides_data_location(tmp_path):
    _, manifest = _make_bundle(tmp_path)
    context = _context(
        run_config={"manifest-path": "wrong.csv", "root-dir": "wrong"},
        node_config={"manifest-path": str(manifest), "data-root": str(tmp_path)},
    )
    config = detection_config_from_context(context)
    assert config.manifest_path == str(manifest)
    assert config.root_dir == str(tmp_path)


def test_env_supplies_data_location_for_clientapp(tmp_path, monkeypatch):
    _, manifest = _make_bundle(tmp_path)
    monkeypatch.setenv("FL_MANIFEST_PATH", str(manifest))
    monkeypatch.setenv("FL_DATA_ROOT", str(tmp_path))

    config = detection_config_from_context(_context())

    assert config.manifest_path == str(manifest)
    assert config.root_dir == str(tmp_path)


def test_node_config_overrides_env_data_location(tmp_path, monkeypatch):
    _, manifest = _make_bundle(tmp_path)
    monkeypatch.setenv("FL_MANIFEST_PATH", "env.csv")
    monkeypatch.setenv("FL_DATA_ROOT", "env-root")
    context = _context(
        node_config={"manifest-path": str(manifest), "root-dir": str(tmp_path)}
    )

    config = detection_config_from_context(context)

    assert config.manifest_path == str(manifest)
    assert config.root_dir == str(tmp_path)


def test_env_client_id_selects_matching_client(tmp_path, monkeypatch):
    bundle, manifest = _make_bundle(tmp_path)
    monkeypatch.setenv("FL_MANIFEST_PATH", str(manifest))
    monkeypatch.setenv("FL_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("FL_CLIENT_ID", "site-b")
    config = detection_config_from_context(_context())

    client = select_detection_client(_context(), bundle, client_id=config.client_id)

    assert client.client_label == "site-b"


def test_server_round_config_contains_detection_hyperparameters():
    config = DetectionConfig(batch_size=1, lr=0.123, pretrained=False)
    record = _round_config(config)
    assert record["batch_size"] == 1
    assert record["lr"] == 0.123
    assert record["score_threshold"] == config.score_threshold


def test_build_strategy_defaults_to_strict_num_clients(monkeypatch):
    captured = {}

    def _fake_fedavg(**kwargs):
        captured.update(kwargs)
        return "STRATEGY"

    monkeypatch.setattr(serverapp, "FedAvg", _fake_fedavg)
    strategy = _build_strategy(DetectionConfig(num_clients=2, pretrained=False))

    assert strategy == "STRATEGY"
    assert captured["min_train_nodes"] == 2
    assert captured["min_evaluate_nodes"] == 2
    assert captured["min_available_nodes"] == 2
    assert captured["fraction_train"] == 1.0
    assert captured["fraction_evaluate"] == 1.0


def test_clientapp_and_serverapp_wired_to_detection_task():
    import src.fl.detection_clientapp as clientapp
    import src.fl.detection_serverapp as serverapp_module
    import src.fl.detection_task as detection_task

    assert isinstance(clientapp._TASK, detection_task.DetectionTask)
    assert isinstance(serverapp_module._TASK, detection_task.DetectionTask)
    # re-exported helpers are the very same objects (backward-compatible imports)
    assert clientapp.detection_config_from_context is detection_task.detection_config_from_context
    assert clientapp.select_detection_client is detection_task.select_detection_client
    assert clientapp.load_detection_client_context is detection_task.load_detection_client_context


def test_build_strategy_uses_explicit_min_node_overrides(monkeypatch):
    captured = {}

    monkeypatch.setattr(serverapp, "FedAvg", lambda **kwargs: captured.update(kwargs))
    _build_strategy(
        DetectionConfig(
            num_clients=2,
            min_train_nodes=1,
            min_evaluate_nodes=1,
            min_available_nodes=1,
            pretrained=False,
        )
    )

    assert captured["min_train_nodes"] == 1
    assert captured["min_evaluate_nodes"] == 1
    assert captured["min_available_nodes"] == 1
